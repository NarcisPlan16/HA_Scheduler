# Optimal Scheduler - Home Assistant Add-on

## Description

The Optimal Scheduler is a Home Assistant add-on designed to optimize energy management in smart grids. By utilizing artificial intelligence techniques and advanced optimization, this add-on helps reduce energy costs by intelligently managing consumption and generation.

With the Optimal Scheduler, you can:
- Optimize energy consumption and generation according to market rates and prices.
- Control consumption and generation devices, including batteries, solar panels, and other energy sources.
- Simulate different scenarios to find the best configuration for your energy community.

## Key Features

- **Intelligent Demand and Generation Management**: Based on machine learning algorithms to adjust consumption according to market conditions and available energy.
- **Support for Multiple Devices**: Control energy devices such as batteries, appliances, generators, and more.
- **Cost Optimization**: Reduce energy costs with advanced scheduling and intelligent use of time-based rates.
- **Scenario Simulation**: Test different configurations and parameters to find the best solution before applying it in real-time.

## How to Use the Add-on

### Installation

1. Download and install the *Optimal Scheduler* add-on from the Supervisor section of Home Assistant.
2. Configure basic parameters in the `config.yaml` file. Example configuration:

    ```yaml
    options:
      Controllable_Consumer_asset_IDs: []
      Controllable_Generator_asset_IDs: []
      Controllable_Energy_Source_asset_IDs: []
      Base_Building_consumption_IDs: []
      Base_Building_generation_IDs: []
      Simulation_code_directory: /config/OptimalScheduler/MySimulationCode
      Classes_code_directory: /config/OptimalScheduler/MyClassesCode
    ```

3. Make sure to specify the identifiers of the devices you want to control, as well as the paths to your simulation and class codes.

### Configuration Parameters

- **Controllable_Consumer_asset_IDs**: List of IDs for the consumable devices that can be controlled.
- **Controllable_Generator_asset_IDs**: List of IDs for the generators that can be controlled.
- **Controllable_Energy_Source_asset_IDs**: List of energy sources such as batteries or solar panels.
- **Base_Building_consumption_IDs**: Identifiers for the fixed consumption of your building.
- **Base_Building_generation_IDs**: Identifiers for the fixed generation (e.g., solar panels).
- **Simulation_code_directory**: Path to the directory containing your simulation code.
- **Classes_code_directory**: Path to the directory containing classes for simulation.

### Usage Examples

Once configured, the Optimal Scheduler will intelligently manage your devices and energy sources according to the configuration you have set. You can view the results in real-time through the Home Assistant dashboard.

### Requirements

- Home Assistant 2023.5 or higher.
- Compatible consumption or generation devices with Home Assistant.

### Support

If you have questions or need help, visit our official repository on [GitHub](https://github.com/NarcisPlan16/HA_Scheduler) or consult the documentation.