{
  description = "A concise Python development environment using uv";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    nixpkgs.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        # Defines the shell that `nix develop` will use.
        devShells.default = pkgs.mkShell {
          # Tools available in the shell.
          buildInputs = [
            pkgs.python3
            pkgs.uv
          ];

          # Code to run when entering the shell.
          shellHook = ''
            # Set the virtual environment to a local .venv directory.
            export VIRTUAL_ENV=$(pwd)/.venv
            # Add the venv's scripts to the PATH.
            export PATH="$VIRTUAL_ENV/bin:$PATH"
            
            # `uv pip sync` ensures the venv matches requirements.txt.
            # It's fast and creates the venv on the first run.
            uv pip sync requirements.txt
            
            echo "Python environment ready."
          '';
        };
      });
}
