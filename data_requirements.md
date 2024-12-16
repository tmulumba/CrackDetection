# Data Requirements for Rock Mechanics Analysis

## Core Data Types

### Image/Video Data (Primary)
- High-resolution images from hydraulic fracturing experiments
- High-speed video recordings
- Required metadata:
  - Frame rate (particularly for Li 2019 and Gonçalves 2016 high-speed videos)
  - Image resolution
  - Time stamps
  - Camera specifications and setup
  - Lighting conditions
  - Field of view relative to specimen

### Acoustic Emissions Data (Secondary)
Required format for all AE data:
- Time of event
- Coordinates (x, y, z)
- Event type (tensile, shear, compressive)
- Magnitude

### Independent Variables
- Specimen characteristics:
  - Rock type (granite vs. clayshale)
  - Bedding plane orientation (degrees)
  - Layer composition (light vs. dark layers in shale)
  - Specimen dimensions (height, width, thickness, area)
  - Flaw dimensions (length, width)
- Loading conditions:
  - Vertical load
  - Water pressure applied to flaws
  - Volume of water injected
  - Flaw geometry and orientation

### Controlled Variables
- Testing conditions:
  - Hydraulic fracturing liquid type and viscosity
- Material properties:
  - Stiffness
  - Hardness
  - Creep compliance
  - Toughness
  - Radius of plasticity
  - Fracture energy
  - Seismic properties:
    - P-wave arrival times
    - Velocity
- Standard specimen dimensions (when specified):
  - 6" x 3" x 1" (Gonçalves 2016)
  - Flaw dimensions: 12.7mm (0.5") long, ~0.7mm wide (Gonçalves 2016)

## Analysis Requirements

### 1. Crack Detection

#### Image Data Requirements
- Complete sequence of images showing crack initiation and propagation
- Images at different stages of loading
- Clear visibility of:
  - Initial flaws
  - Developing cracks
  - Material boundaries
  - Bedding planes

#### Material Context
- Rock type specifications:
  - Opalinus Shale (AlDajani 2022)
  - Barre Granite (Li 2019)
  - Vaca Muerta Shale (Gonçalves 2016)
- Specimen dimensions
- Bedding plane orientations
- Layer composition (for shales)

#### Proposed Analysis
- Training data creation:
  - Manual labeling of cracks in subset of images
  - Use of existing crack sketches from AlDajani 2022
- Model validation:
  - Cross-validation across different rock types
  - Performance evaluation on both shale and granite
- Segmentation model application:
  - Initial testing with pre-trained models
  - Fine-tuning for specific rock types
  - Performance comparison across different materials

### 2. Crack Classification

#### Primary Data Requirements
- Processed images from crack detection stage
- Strain field data:
  - DIC data where available (Li 2019)
  - Strain-Net generated fields for non-speckled images
- AE data with clear event classifications:
  - Temporal correlation with images
  - Spatial correlation with visible cracks
  - Event type classification (tensile/shear/compressive)

#### Supporting Data
- Loading conditions:
  - Applied loads and displacements
  - Internal flaw pressure
  - Injection rates
- Material properties:
  - Stiffness
  - Hardness
  - Creep compliance
  - Toughness
  - Seismic properties

#### Proposed Analysis
- Strain field analysis:
  - Computation of strain ratios
  - Principal strain calculation
  - Strain orientation analysis
- Machine learning classification:
  - Feature extraction from strain fields
  - Integration of AE data
  - Model training and validation
- Cross-validation:
  - Testing across different rock types
  - Validation against AE event classifications

### 3. Crack Propagation

#### Time Series Data Requirements
- Temporal sequence of crack positions
- Time-stamped images/frames
- Loading history:
  - Injection rates
  - Pressure measurements
  - Applied loads
- AE event sequence:
  - Event timing
  - Location progression
  - Event types and magnitudes

#### Environmental Variables
- Specimen conditions:
  - Dimensions
  - Initial flaw geometry
  - Bedding plane orientation
- Testing conditions:
  - Loading rates
  - Fluid properties
  - Boundary conditions

#### Proposed Analysis
- Time series modeling:
  - ARIMA/ARIMAX baseline models
  - Integration of spatial and temporal features
- Deep learning approach:
  - CNN for spatial feature extraction
  - RNN for temporal prediction
  - Hybrid model development
- Validation:
  - Cross-validation across different experiments
  - Performance comparison with classical models
  - Error analysis and model refinement

## Data Availability by Thesis

### AlDajani 2022: Hydraulic Fracturing Behavior of Opalinus Shale: A Framework, Experimentation & Insights
- Primary focus: Opalinus Shale
- Key data: High-resolution images, AE data, detailed material properties

### Li 2019: Microseismic and real-time imaging of fractures and microfractures in barre granite and opalinus clayshale.
- Primary focus: Barre Granite and Opalinus Clayshale
- Key data: High-speed video, detailed AE analysis, DIC measurements

### Gonçalves 2016: Fracturing processes and induced seismicity due to the hydraulic fracturing of rocks.
- Primary focus: Barre Granite, Vaca Muerta shale, Opalinus Shale
- Key data: AE waveforms, high-speed video
