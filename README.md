# AGAMA initial conditions for isolated disc galaxies

Scripts for building equilibrium initial conditions (stellar disc + live dark-matter halo,
optionally a gas disc) with [AGAMA](https://github.com/GalacticDynamics-Oxford/AGAMA)
(Vasiliev 2019, MNRAS 482, 1525), and writing them as Gadget HDF5 files.

## Pipeline

```
make_AGAMA_ini.py    ->  ini/AGAMA_C<C>_MD<MD>.ini        model parameters (grid in C = Rh/Rd and MD = Mdisk/Mhalo)
run_AGAMA_model.py   ->  model_{stars,dm[,gas]}_final_<tag>.txt  + model_final_<tag>.hdf5 (--hdf5)
check_gadget_run.py      stability diagnostics of a Gadget-4 run (halo/disc radii, thickness, dispersions, A2, energy)
plot_projection_grid.py  face-on / edge-on projections of every snapshot (--ptype 0 for gas)
```

* **Disc:** `QuasiIsothermal` DF; **halo:** isotropic `QuasiSpherical` DF reproducing a Hernquist profile.
  Both are iterated to self-consistency in the total potential.
* **Gas (optional, `gas_disc.py`):** isothermal exponential disc in vertical hydrostatic equilibrium,
  included in the self-consistent potential; `u = c_s^2` for Gadget-4 `ISOTHERM_EQS`.
* HDF5 output is point-symmetrically mirrored (`write_mirror_particles.py`); masses default to Msun,
  matching gadget4 `UnitMass_in_g = 1.988e33`.

## Example

```bash
cd ini && python ../make_AGAMA_ini.py && cd ..
python run_AGAMA_model.py ini/AGAMA_C7p5_MD0p02.ini --ndisk 1e6 --nhalo 1e7 --hdf5
python run_AGAMA_model.py ini/AGAMA_C7p5_MD0p02.ini --gas-frac 0.1 --gas-cs 10 --ngas 1e6 --hdf5 --tag gas0p1
```

## Installing AGAMA (macOS, Homebrew)

```bash
brew install gsl
CPATH=/opt/homebrew/include LIBRARY_PATH=/opt/homebrew/lib \
  pip install --no-build-isolation agama --config-settings --build-option=--yes
```

Notes for AGAMA 1.0: `outerCutoffRadius` is not accepted for `type=Dehnen` (dropped by default;
`--halo-cutoff` applies it via a truncated `Spheroid`).
