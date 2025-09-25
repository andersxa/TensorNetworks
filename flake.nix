{
  description = "ttbregman shell";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs = {
    self,
    nixpkgs,
    ...
  }: let
    pkgs = nixpkgs.legacyPackages."x86_64-linux";
    python = pkgs.python3;

    # Define the custom Python package separately
    ucimlrepo = python.pkgs.buildPythonPackage rec {
      pname = "ucimlrepo";
      version = "0.0.7";
      src = pkgs.fetchFromGitHub {
        owner = "uci-ml-repo";
        repo = "ucimlrepo";
        rev = "main";
        # sha256 = nixpkgs.lib.fakeHash;
        sha256 = "sha256-5R4/edQriufhVU1UXCY7nTfEdwRhi33e/CHdTkLf3jo=";
      };
      # Add dependencies if needed
      # propagatedBuildInputs = with python.pkgs; [ ... ];
    };
  in {
    devShells.x86_64-linux.default = pkgs.mkShell {
      packages = [
        (python.withPackages (p:
          with p; [
            seaborn
            matplotlib
            tqdm
            torch
            jedi-language-server
            black
            pandas
            ucimlrepo
            scikit-learn
          ]))
      ];
    };
  };
}
