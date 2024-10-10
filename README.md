# ExitOS Add-on Repository for Home Assistant

This repository contains a custom Home Assistant add-on, **ExitOS**, designed for managing smart energy networks. The add-on enables the configuration, simulation, and management of various energy assets, such as buildings, consumers, generators, and energy sources.

## Add-on documentation: <https://developers.home-assistant.io/docs/add-ons>

[![Open your Home Assistant instance and show the add add-on repository dialog with a specific repository URL pre-filled.](https://my.home-assistant.io/badges/supervisor_add_addon_repository.svg)](https://my.home-assistant.io/redirect/supervisor_add_addon_repository/?repository_url=https://github.com/NarcisPlan16/HA_Scheduler)

## Add-ons

This repository contains the following add-ons:

### [Exit OS](./exit_os)

![Supports aarch64 Architecture][aarch64-shield]
![Supports amd64 Architecture][amd64-shield]
![Supports armhf Architecture][armhf-shield]
![Supports armv7 Architecture][armv7-shield]
![Supports i386 Architecture][i386-shield]

_Exit OS manages a smart energy network by integrating various energy assets, such as buildings, consumers, and generators. It allows users to simulate energy production and consumption, making it ideal for energy communities._

### Features

- **Asset Configuration**: Allows you to configure various energy assets using `.toml` files.
- **Asset Simulation**: Provides the ability to simulate the performance of energy assets, including energy generation and consumption.
- **Dynamic Asset Management**: Easily manage buildings, generators, consumers, and energy sources.
- **Controllable Assets**: Supports integration of controllable energy sources.
- **Supports Multiple Architectures**: Compatible with a variety of hardware platforms.

### Usage Instructions

1. Add this repository to Home Assistant.
2. Configure your energy assets using the configuration files.
3. Run simulations to manage and monitor energy production and consumption.
4. Manage the system through the provided web interface.

For further information on how to configure assets or to contribute to the development of this add-on, check the code and documentation in this repository.

---

[aarch64-shield]: https://img.shields.io/badge/aarch64-yes-green.svg
[amd64-shield]: https://img.shields.io/badge/amd64-yes-green.svg
[armhf-shield]: https://img.shields.io/badge/armhf-yes-green.svg
[armv7-shield]: https://img.shields.io/badge/armv7-yes-green.svg
[i386-shield]: https://img.shields.io/badge/i386-yes-green.svg



<!--# Example Home Assistant add-on repository

This repository can be used as a "blueprint" for add-on development to help you get started.

Add-on documentation: <https://developers.home-assistant.io/docs/add-ons>

[![Open your Home Assistant instance and show the add add-on repository dialog with a specific repository URL pre-filled.](https://my.home-assistant.io/badges/supervisor_add_addon_repository.svg)](https://my.home-assistant.io/redirect/supervisor_add_addon_repository/?repository_url=https://github.com/NarcisPlan16/HA_Scheduler)

## Add-ons

This repository contains the following add-ons

### [Example add-on](./example)

![Supports aarch64 Architecture][aarch64-shield]
![Supports amd64 Architecture][amd64-shield]
![Supports armhf Architecture][armhf-shield]
![Supports armv7 Architecture][armv7-shield]
![Supports i386 Architecture][i386-shield]

_Example add-on to use as a blueprint for new add-ons._

Notes to developers after forking or using the github template feature:
- While developing comment out the 'image' key from 'example/config.yaml' to make the supervisor build the addon
  - Remember to put this back when pushing up your changes.
- When you merge to the 'main' branch of your repository a new build will be triggered.
  - Make sure you adjust the 'version' key in 'example/config.yaml' when you do that.
  - Make sure you update 'example/CHANGELOG.md' when you do that.
  - The first time this runs you might need to adjust the image configuration on github container registry to make it public
  - You may also need to adjust the github Actions configuration (Settings > Actions > General > Workflow > Read & Write)
- Adjust the 'image' key in 'example/config.yaml' so it points to your username instead of 'home-assistant'.
  - This is where the build images will be published to.
- Rename the example directory.
  - The 'slug' key in 'example/config.yaml' should match the directory name.
- Adjust all keys/url's that points to 'home-assistant' to now point to your user/fork.
- Share your repository on the forums https://community.home-assistant.io/c/projects/9
- Do awesome stuff!

[aarch64-shield]: https://img.shields.io/badge/aarch64-yes-green.svg
[amd64-shield]: https://img.shields.io/badge/amd64-yes-green.svg
[armhf-shield]: https://img.shields.io/badge/armhf-yes-green.svg
[armv7-shield]: https://img.shields.io/badge/armv7-yes-green.svg
[i386-shield]: https://img.shields.io/badge/i386-yes-green.svg-->