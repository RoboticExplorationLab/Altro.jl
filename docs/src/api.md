# Altro.jl API 
```@meta
CurrentModule = Altro
```

```@contents
Pages = ["api.md"]
```

This page provides the docstrings for the most common methods that the user may
work with.

## Solvers

### Types
```@docs
ALTROSolver
AugmentedLagrangianSolver
iLQRSolver
ProjectedNewtonSolver
```

### Methods
```@docs
set_options!
backwardpass!
forwardpass!
record_iteration!
```

```