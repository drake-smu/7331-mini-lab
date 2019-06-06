let
  jupyter = import (builtins.fetchGit {
   url = https://github.com/tweag/jupyterWith;
    rev = "";
  }) {};

  iPython = jupyter.kernels.iPythonWith {
    name = "python";
    packages = p: with p; [
      plotly
      numpy
      pandas
      jupytext
      matplotlib
      scikitlearn
      seaborn
      scipy
      future
      ipywidgets
      scikitimage
      tzlocal
      simplegeneric
      pprintpp
    ];
  };

  iR = jupyter.kernels.juniperWith {
    name = "R";
    packages = p: with p; [
       mlbench
       arules
       tidyverse
       ggplot2
       dplyr
    ];
  };

  iHaskell = jupyter.kernels.iHaskellWith {
    name = "haskell";
    packages = p: with p; [ hvega formatting ];
  };

  jupyterEnvironment =
    jupyter.jupyterlabWith {
      kernels = [ iPython iHaskell iR];
#      directory = jupyter.mkDirectoryWith {
#        extensions = [
#         "jupyterlab-jupytext"
#        ];
#      };
    };
    pkgs = import <nixpkgs> {};
in
  pkgs.mkShell {
    name = "environment";
    buildInputs = with pkgs; [
      vscode
       jupyterEnvironment
       python37
       python37Packages.pandas
       python37Packages.numpy
       python37Packages.matplotlib
       #python37Packages.sqlite
       python37Packages.notebook
       python37Packages.ipython
       python37Packages.jupytext
       python37Packages.scikitlearn
       python37Packages.seaborn
       python37Packages.scipy
       python37Packages.plotly
       python37Packages.ipywidgets
       python37Packages.future
       python37Packages.scikitimage
       #Todo Package graphlab
       python37Packages.tzlocal
       rstudio
       python37Packages.simplegeneric
       R
       rPackages.mlbench
       rPackages.lobstr
       rPackages.lubridate
       rPackages.stringr
       rPackages.abind
       rPackages.foreign
       rPackages.downloader
       rPackages.memoise
       rPackages.lattice
       rPackages.microbenchmark
       rPackages.arules
       rPackages.tidyverse
       rPackages.devtools
       rPackages.pander
       rPackages.Rcpp
       rPackages.RNHANES
       rPackages.reticulate
       rPackages.humaniformat
       python37Packages.pprintpp
    ];
    
  }
