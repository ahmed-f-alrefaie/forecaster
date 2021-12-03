forecaster
==========

Forecaster uses a probabilistic mass-radius relation as the underlying model.

It can forecast mass (radius) given radius (mass) measurements.

The conversion includes three sources of uncertainties, from measurement, model fitting (MCMC process), and intrinsic dispersion in radius.

See arXiv:1603.08614 for details. 

If you use it, please cite it.

Changes
-------

This fork introduces some changes to  make it easier to use and improve performance. One major change is with the
*piece_linear* code which has been vectorized. Running the old version using 10000 samples takes about *5.38 seconds*,
whilst the new version taks *2.44 milliseconds* which is a 2200x speed up making it useful for realtime applications.


Additionally you can now use it anywhere. Install using
```
pip install git+https://github.com/ahmed-f-alrefaie/forecaster.git
```

Then run from your python terminal:

```python
>>> import forecaster.mr_forecast as mr
>>> mr.Mstat2R(mean=1.0, std=0.1, unit='Earth', sample_size=100000)
(1.006464878768933, 0.11527046644017758, 0.10215885395157043)
```


Usage
-----

Check demo.ipynb for more details.


A simple example:

	import numpy as np
	import mr_forecast as mr
	
	# predict the mean and std of radius given mass measurements

	Rmedian, Rplus, Rminus = mr.Mstat2R(mean=1.0, std=0.1, unit='Earth', sample_size=100)

A simple interactive example:
    
    print '=== Forecaster ==='
    print ' '
    print 'Example: Radius-to-Mass Conversion (without posteriors)'
    print ' '
    print 'Radius = A +/- B [Earth units]'
    mean = float(raw_input("Enter A: "))
    std = float(raw_input("Enter B: "))

    # predict the mean and std of radius given mass measurements
    Mmedian, Mplus, Mminus = mr.Rstat2M(mean, std, unit='Earth', sample_size=1e3, grid_size=1e3)
    print ' '
    print 'Mass = ',Mmedian,'+',Mplus,'-',Mminus,' M_earth'



