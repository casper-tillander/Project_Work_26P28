
## Build in VS Code, nRF Connect

1. Open `embedded/nrf5340_regressor_app` as an nRF Connect application.
2. Add build configuration.
3. Select board: `nrf5340dk_nrf5340_cpuapp`.
4. Build.
5. Flash.
6. Open serial log / terminal.

Expected output starts with:

```text
Regressor embedded test start
```

The printed predictions should match the expected values. The exported model is large. If the build fails because of flash size, retrain a smaller model, for example with fewer trees and a smaller max depth, then run the export scripts again and copy the new `rf_regressor_model.c/.h` into this folder.
