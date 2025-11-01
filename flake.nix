{
  description = "A concise Python development environment using uv";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      # Explicitly define the systems you want to support.
      supportedSystems = [ "x86_64-linux" "aarch64-darwin" "x86_64-darwin" ];

      # A function to generate the dev shell for a given system's pkgs.
      perSystem = pkgs: {
        devShells.default = pkgs.mkShell {
          # Tools available in the shell.
          buildInputs = [
            pkgs.python3
            pkgs.uv
          ];

          # Code to run when entering the shell.
          shellHook = ''
            echo "Python environment ready."
            source .venv/bin/activate
          '';
        };
      };

    in
    # This block iterates over `supportedSystems` and builds the outputs.
    nixpkgs.lib.foldl' (final: system:
      nixpkgs.lib.recursiveUpdate final {
        devShells.${system} = (perSystem nixpkgs.legacyPackages.${system}).devShells;
      })
      { }
      supportedSystems;
}


