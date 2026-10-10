{
  description = "prosper development shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      devShells = forAllSystems (
        pkgs:
        let
          vulkanSDK = pkgs.symlinkJoin {
            name = "vulkan-sdk";
            paths = [
              pkgs.vulkan-headers
              pkgs.vulkan-loader
              pkgs.vulkan-validation-layers
            ];
          };
        in
        {
          default = pkgs.mkShell {
            name = "prosper";

            packages = with pkgs; [
              cmake
              ninja
              gcc
              binutils
              gdb
              git
              pkg-config
              python3
              ccache
              cacert
              spirv-tools
              vulkan-validation-layers
              mesa
              ffmpeg_7
              libva
              sdl3
              zlib
              clang-tools
              ruff
            ];

            hardeningDisable = [ "format" "fortify" "fortify3" ];

            SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";

            shellHook = ''
              export VULKAN_SDK="${vulkanSDK}"
              export VK_LAYER_PATH="${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d"
              export CCACHE_SLOPPINESS="pch_defines,time_macros"

              if [ -z "''${VK_ICD_FILENAMES:-}''${VK_DRIVER_FILES:-}" ] \
                 && ! ls /run/opengl-driver/share/vulkan/icd.d/*.json >/dev/null 2>&1 \
                 && ! ls /usr/share/vulkan/icd.d/*.json >/dev/null 2>&1; then
                export VK_DRIVER_FILES="${pkgs.mesa}/share/vulkan/icd.d/lvp_icd.x86_64.json"
              fi

              echo "prosper devshell"
              echo "  cmake -S prosper -B prosper/build-linux -G Ninja -DPROSPER_APP=ON -DPROSPER_AUDIO_SDL3=ON -DPROSPER_PAD_SDL3=ON"
              echo "  cmake --build prosper/build-linux"
              echo "  ctest --test-dir prosper/build-linux --no-tests=error"
            '';
          };
        }
      );
    };
}
