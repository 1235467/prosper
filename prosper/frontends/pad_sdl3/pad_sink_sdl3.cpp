// pad_sink_sdl3.cpp — SDL3 SDL_Gamepad PadBackend (optional frontend, -DPROSPER_PAD_SDL3=ON).
//
// Bridges the headless PadBackend interface (src/input/pad.hpp) to SDL3's cross-platform gamepad
// API. SDL maps any recognised controller (DualSense, Xbox, Switch Pro, …) to a canonical layout
// via its bundled mapping database, so there is no per-device button/axis guesswork here — SDL's
// canonical buttons/axes map 1:1 to the PS5 layout. One gamepad handle per pad index, opened lazily.
#include "pad_sdl3.hpp"
#include "../../src/input/pad.hpp"

#include <SDL3/SDL.h>

#include <array>
#include <chrono>
#include <cstring>
#include <mutex>

namespace prosper {
namespace {

using namespace prosper::input;

constexpr int kMaxPads = 4;

// Sony 0..255 from an SDL stick axis (-32768..32767) or trigger axis (0..32767 -> rests at 0).
uint8_t stick_u8(Sint16 v)   { return pad_axis_u8((int)v, -32768, 32767); }
uint8_t trigger_u8(Sint16 v) { return pad_axis_u8((int)v, 0, 32767); }

// SDL canonical gamepad button -> Sony bit (0 = no mapping).
uint32_t sony_button_for(SDL_GamepadButton b) {
    switch (b) {
        case SDL_GAMEPAD_BUTTON_SOUTH:          return SCE_PAD_BUTTON_CROSS;
        case SDL_GAMEPAD_BUTTON_EAST:           return SCE_PAD_BUTTON_CIRCLE;
        case SDL_GAMEPAD_BUTTON_NORTH:          return SCE_PAD_BUTTON_TRIANGLE;
        case SDL_GAMEPAD_BUTTON_WEST:           return SCE_PAD_BUTTON_SQUARE;
        case SDL_GAMEPAD_BUTTON_START:          return SCE_PAD_BUTTON_OPTIONS;
        case SDL_GAMEPAD_BUTTON_LEFT_STICK:     return SCE_PAD_BUTTON_L3;
        case SDL_GAMEPAD_BUTTON_RIGHT_STICK:    return SCE_PAD_BUTTON_R3;
        case SDL_GAMEPAD_BUTTON_LEFT_SHOULDER:  return SCE_PAD_BUTTON_L1;
        case SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER: return SCE_PAD_BUTTON_R1;
        case SDL_GAMEPAD_BUTTON_DPAD_UP:        return SCE_PAD_BUTTON_UP;
        case SDL_GAMEPAD_BUTTON_DPAD_DOWN:      return SCE_PAD_BUTTON_DOWN;
        case SDL_GAMEPAD_BUTTON_DPAD_LEFT:      return SCE_PAD_BUTTON_LEFT;
        case SDL_GAMEPAD_BUTTON_DPAD_RIGHT:     return SCE_PAD_BUTTON_RIGHT;
        case SDL_GAMEPAD_BUTTON_TOUCHPAD:       return SCE_PAD_BUTTON_TOUCH_PAD;
        default:                                return 0;   // BACK/GUIDE/paddles/misc: no PS5 bit
    }
}

class Sdl3PadBackend : public PadBackend {
public:
    bool init() {
        if (!SDL_InitSubSystem(SDL_INIT_GAMEPAD)) {
            SDL_Log("prosper-pad: SDL_InitSubSystem(GAMEPAD) failed: %s", SDL_GetError());
            return false;
        }
        return true;
    }
    void quit() {
        std::lock_guard<std::mutex> lk(mx_);
        for (auto& g : gp_) { if (g) { SDL_CloseGamepad(g); g = nullptr; } }
        SDL_QuitSubSystem(SDL_INIT_GAMEPAD);
    }

    bool poll(int index, HostPadState& out) override {
        out = HostPadState{};
        if (index < 0 || index >= kMaxPads) return false;

        std::lock_guard<std::mutex> lk(mx_);
        SDL_UpdateGamepads();                    // refresh state (we don't run an SDL event loop)

        SDL_Gamepad* g = gamepad(index);
        if (!g) return false;                    // no device at this index
        refresh_rumble(g);

        uint32_t b = 0;
        for (int i = 0; i < SDL_GAMEPAD_BUTTON_COUNT; i++)
            if (SDL_GetGamepadButton(g, (SDL_GamepadButton)i))
                b |= sony_button_for((SDL_GamepadButton)i);

        out.left_x  = stick_u8(SDL_GetGamepadAxis(g, SDL_GAMEPAD_AXIS_LEFTX));
        out.left_y  = stick_u8(SDL_GetGamepadAxis(g, SDL_GAMEPAD_AXIS_LEFTY));
        out.right_x = stick_u8(SDL_GetGamepadAxis(g, SDL_GAMEPAD_AXIS_RIGHTX));
        out.right_y = stick_u8(SDL_GetGamepadAxis(g, SDL_GAMEPAD_AXIS_RIGHTY));
        out.l2      = trigger_u8(SDL_GetGamepadAxis(g, SDL_GAMEPAD_AXIS_LEFT_TRIGGER));
        out.r2      = trigger_u8(SDL_GetGamepadAxis(g, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER));
        b |= pad_trigger_buttons(out.l2, out.r2);

        out.buttons   = b;
        out.connected = true;
        return true;
    }

    // Effector output from the libScePad HLE: vibration (SDL rumble) and, on a DualSense, the
    // adaptive-trigger effects as SDL_SendGamepadEffect's 47-byte effects payload. SDL prepends the
    // report id (0x02 USB / 0x31 Bluetooth) and the Bluetooth framing, so the payload is exactly the
    // DS5EffectsState_t body: enable byte, the motors, then the R2/L2 11-byte effects at 10/21.
    void apply_output(int index, const HostPadOutput& out) override {
        if (index < 0 || index >= kMaxPads) return;
        std::lock_guard<std::mutex> lk(mx_);
        SDL_Gamepad* g = gamepad(index);
        if (!g) return;
        output_ = out;
        send_effects(g, out);
    }

private:
    static constexpr Uint32 kRumbleMs = 2000;   // SDL rumble auto-expires; refresh_rumble re-arms it
    static constexpr auto kRumbleRefresh = std::chrono::milliseconds(700);

    // Open (and hot-plug-track) the index-th currently-connected gamepad, or nullptr. mx_ held.
    SDL_Gamepad* gamepad(int index) {   // NOLINT(readability-make-member-function-const): opens lazily
        if (index < 0 || index >= kMaxPads) return nullptr;
        SDL_Gamepad*& g = gp_[index];
        if (g && !SDL_GamepadConnected(g)) { SDL_CloseGamepad(g); g = nullptr; }   // hot-unplug
        if (!g) g = open_index(index);
        return g;
    }

    void send_effects(SDL_Gamepad* g, const HostPadOutput& out) {   // mx_ held
        const bool rumbling = out.vibration_large != 0 || out.vibration_small != 0;
        SDL_RumbleGamepad(g, (Uint16)(out.vibration_large * 257), (Uint16)(out.vibration_small * 257),
                          rumbling ? kRumbleMs : 0);
        last_rumble_ = std::chrono::steady_clock::now();
        if (!out.trigger_valid) return;
        if (SDL_GetGamepadType(g) == SDL_GAMEPAD_TYPE_PS5) {
            // 47-byte DS5 effects payload: 0x04 enables the R2 effect, 0x08 the L2 one.
            Uint8 effect[47] = {};
            effect[0] = 0x04 | 0x08;
            std::memcpy(effect + 10, out.trigger_effect[1], 11);   // R2
            std::memcpy(effect + 21, out.trigger_effect[0], 11);   // L2
            SDL_SendGamepadEffect(g, effect, (int)sizeof(effect));
        } else {
            // Pads without adaptive triggers get the effect's strength as ordinary trigger rumble
            // (SDL returns false when the device has none; that is the documented no-op).
            const bool trigger_rumble = out.trigger_rumble[0] != 0 || out.trigger_rumble[1] != 0;
            SDL_RumbleGamepadTriggers(g, (Uint16)(out.trigger_rumble[0] * 257),
                                      (Uint16)(out.trigger_rumble[1] * 257), trigger_rumble ? kRumbleMs : 0);
        }
    }

    void refresh_rumble(SDL_Gamepad* g) {   // mx_ held; poll path
        if (output_.vibration_large == 0 && output_.vibration_small == 0) return;
        const auto now = std::chrono::steady_clock::now();
        if (now - last_rumble_ < kRumbleRefresh) return;
        SDL_RumbleGamepad(g, (Uint16)(output_.vibration_large * 257),
                          (Uint16)(output_.vibration_small * 257), kRumbleMs);
        last_rumble_ = now;
    }

    // Open the index-th currently-connected gamepad (SDL instance-id order), or nullptr.
    SDL_Gamepad* open_index(int index) {
        int count = 0;
        SDL_JoystickID* ids = SDL_GetGamepads(&count);
        SDL_Gamepad* g = nullptr;
        if (ids && index < count) g = SDL_OpenGamepad(ids[index]);
        SDL_free(ids);
        return g;
    }
    std::mutex mx_;
    std::array<SDL_Gamepad*, kMaxPads> gp_{};
    HostPadOutput output_;
    std::chrono::steady_clock::time_point last_rumble_{};
};

Sdl3PadBackend g_backend;
bool           g_installed = false;

} // namespace

bool install_sdl3_pad_backend() {
    if (g_installed) return true;
    if (!g_backend.init()) return false;
    input::pad_set_backend(&g_backend);
    g_installed = true;
    SDL_Log("prosper-pad: SDL3 gamepad backend installed");
    return true;
}

void shutdown_sdl3_pad_backend() {
    if (!g_installed) return;
    input::pad_set_backend(nullptr);
    g_backend.quit();
    g_installed = false;
}

} // namespace prosper
