# Build concrete types for information measures for SOPs
struct TauHat <: InformationMeasure end
struct KappaHat <: InformationMeasure end
struct TauTilde <: InformationMeasure end
struct KappaTilde <: InformationMeasure end

# Build Refinement Types
abstract type RefinedType end
struct RotationType <: RefinedType end
struct DirectionType <: RefinedType end
struct DiagonalType <: RefinedType end

