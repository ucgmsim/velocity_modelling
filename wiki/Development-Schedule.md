# NZCVM Development Schedule

This document outlines the development history, current status, and future roadmap for the New Zealand Community Velocity Model (NZCVM).

## Version History

### Version 1.00a (Current)
- Translation of the original [C code](https://github.com/ucgmsim/Velocity-Model).
- Performance improvements over the original C code (using 1 CPU).
- Data files curated and renamed to follow a consistent naming scheme.
- Enhanced performance for loading tomography data using the HDF5 format (25s → 0.4s).
- Support for CSV and HDF5 output formats.
- Automated testing directly comparing outputs with the original C code.
- Comprehensive documentation and tools for generating basin pages.
- Differences from the C code:
    - Each basin can now have multiple boundaries. Previously, basins with multiple boundaries were treated as separate basins (e.g., Napier_1, Napier_2, Napier_3 → Napier).
    - Basin membership for all mesh grid points is pre-processed for speed.
    - `in_basin_mask.b` assigns `-1` for points not in any basin. The original C code incorrectly assigned `0`, which conflicted with the basin ID `0`.

## Current Development

Our team is currently focused on:

- Expanding documentation and examples.
- Extensive testing and verification.
- Submodel plug-and-play functionality.

## Future Roadmap

### Short-term Goals (Next 6-12 months)
- **Version 2.0**:
    - Performance improvements through parallelization.
    - Integration of a web interface for velocity model generation (for small domains).
    - Incorporation of recent community data.

### Medium-term Goals (1-2 years)
- **Version 3.0**:
    - To be determined through community consultation.

### Long-term Vision (2+ years)
- **Version 4.0**:
    - To be determined through community consultation.

## Release Schedule

| Version | Planned Release | Major Features       |
|---------|-----------------|----------------------|
| 1.00    | Q2 2025         | Initial release      |

## How to Contribute

We welcome contributions from the research community to help advance the NZCVM. If you're interested in contributing, please:

1. Review our current issues and development roadmap.
2. Discuss your proposed changes via GitHub issues.
3. Submit pull requests with well-documented code and test cases.

## References

- Thomson, E.M., Bradley, B.A., & Lee, R.L. (2020). Methodology and computational implementation of a New Zealand Velocity Model (nzcvm2.0) for broadband ground motion simulation. *New Zealand Journal of Geology and Geophysics, 63*(1), 110-127.
- Eberhart-Phillips, D., Reyners, M., Bannister, S., Chadwick, M., & Ellis, S. (2010). Establishing a Versatile 3-D Seismic Velocity Model for New Zealand. *Seismological Research Letters, 81*(6), 992-1000.
