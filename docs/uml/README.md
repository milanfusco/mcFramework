# McFramework UML Diagrams

This directory contains UML class diagrams for the `mcframework` package.

## Available Formats

### 1. Mermaid (Markdown)
**File:** `mcframework_class_diagram.md`

Mermaid diagrams render directly in:
- GitHub/GitLab/Bitbucket README viewers
- VS Code with Mermaid extension
- Obsidian, Notion, and other Markdown editors
- [Mermaid Live Editor](https://mermaid.live/)

### 2. PlantUML
**File:** `mcframework.puml`

PlantUML can be rendered using:
- [PlantUML Online Server](https://www.plantuml.com/plantuml/uml/)
- VS Code with PlantUML extension
- IntelliJ IDEA with PlantUML plugin
- CLI: `java -jar plantuml.jar mcframework.puml`

## Generating PNG/SVG Images

### Option 1: Using pyreverse (from pylint)

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Generate class diagrams
pyreverse -o png -p mcframework -d docs/uml/generated src/mcframework/

# Generates:
#   - classes_mcframework.png  (class diagram)
#   - packages_mcframework.png (package diagram)
```

### Option 2: Using PlantUML CLI

```bash
# Install PlantUML (macOS)
brew install plantuml

# Generate PNG
plantuml docs/uml/mcframework.puml

# Generate SVG
plantuml -tsvg docs/uml/mcframework.puml
```

### Option 3: Using Mermaid CLI

```bash
# Install mermaid-cli
npm install -g @mermaid-js/mermaid-cli

# Generate PNG from Mermaid markdown
mmdc -i mcframework_class_diagram.md -o mcframework_class_diagram.png
```

## Diagram Overview

### Class Diagram Contents

The UML diagram documents:

1. **Core Module (`mcframework.core`)**
   - `MonteCarloSimulation` - Abstract base class
   - `SimulationResult` - Result container (dataclass)
   - `MonteCarloFramework` - Registry/runner for multiple simulations

2. **Stats Engine Module (`mcframework.stats_engine`)**
   - `StatsEngine` - Metric orchestrator
   - `StatsContext` - Configuration dataclass
   - `ComputeResult` - Results with error tracking
   - `FnMetric` - Metric adapter
   - `Metric` - Protocol for custom metrics
   - Enums: `CIMethod`, `NanPolicy`, `BootstrapMethod`

3. **Simulations Module (`mcframework.sims`)**
   - `PiEstimationSimulation`
   - `PortfolioSimulation`
   - `BlackScholesSimulation`
   - `BlackScholesPathSimulation`

4. **Utils Module (`mcframework.utils`)**
   - `z_crit()`, `t_crit()`, `autocrit()`

### Key Relationships

| Relationship | Description |
|--------------|-------------|
| **Inheritance** | All simulations extend `MonteCarloSimulation` |
| **Composition** | `MonteCarloFramework` owns simulations and results |
| **Dependency** | `MonteCarloSimulation` uses `StatsEngine` and `StatsContext` |
| **Protocol** | `FnMetric` implements `Metric` protocol |

