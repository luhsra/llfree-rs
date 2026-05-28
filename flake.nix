{
  description = "LLZero Benchmark Environment";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-26.05";
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, fenix, ... }:
    let
      supportedSystems =
        [ "aarch64-linux" "x86_64-linux" "aarch64-darwin" "x86_64-darwin" ];
      forEachSupportedSystem = f:
        nixpkgs.lib.genAttrs supportedSystems (system:
          f {
            pkgs = import nixpkgs {
              inherit system;
              overlays = [ self.overlays.default ];
            };
          });
    in {
      # Overlay for the Rust toolchain
      overlays.default = final: prev: {
        rustToolchain =
          fenix.packages.${prev.stdenv.hostPlatform.system}.fromToolchainFile {
            file = ./rust-toolchain.toml;
            sha256 = "sha256-gh/xTkxKHL4eiRXzWv8KP7vfjSk61Iq48x47BEDFgfk=";
          };
      };

      # Inputs for shell environments
      devShells = forEachSupportedSystem ({ pkgs }: {
        default = pkgs.mkShell.override { stdenv = pkgs.clangStdenv; } {
          buildInputs = with pkgs;
            [
              rustToolchain
              cargo-edit
              rust-analyzer
              lldb
              curl
              zig
              typst
              python3
              uv
              nodejs
              nix-ld
              libcxx
              libgcc.lib
              zlib
            ];
          env = {
            # Rust-analyzer
            RUST_SRC_PATH =
              "${pkgs.rustToolchain}/lib/rustlib/src/rust/library";
            # Rust Bindgen
            LIBCLANG_PATH = "${pkgs.libclang.lib}/lib/libclang.so";
            LD_LIBRARY_PATH = "${pkgs.libgcc.lib}/lib:${pkgs.zlib}/lib";
          };
        };
      });
    };
}
