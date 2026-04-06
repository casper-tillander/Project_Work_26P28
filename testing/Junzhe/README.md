# Deploying the LDA Classifier on nRF5340-DK

## Platform Note

> **This guide was written and tested on macOS.**  
> The overall workflow is the same on Windows and Linux, but some details—especially the **serial port name**, **USB device path**, and **terminal configuration interface**—may look different.
> 
> For example:
> 
> - **macOS / Linux:** `/dev/tty.usbmodem...` or `ttyACM0`
> - **Windows:** `COM3`, `COM4`, etc.
> 
> The **build, flash, and serial monitoring steps remain the same**, but the exact port selection UI may vary depending on your operating system and installed drivers.

## Step 1: Create the Project Structure

The easiest way to get started is by duplicating the standard Zephyr **Hello World** template, which automatically sets up the build system.

1. Open **VS Code** with the **nRF Connect for VS Code** extension installed.
    
2. Open the **nRF Connect** panel from the left sidebar.
    
3. Click **Create a New Application**.
    
4. Select **Copy a sample**, search for `hello_world`, and choose it.
    
5. Name your project (for example, `nrf5340_lda_classifier`) and select a folder.
    
6. Open the newly created project folder.
    

---

## Step 2: Add the Header Files

Inside your project folder, locate the `src/` directory.

Copy the following files into `src/`, alongside `main.c`:

- `model_lda.h`
    
- `test_data.h`
    

Your folder structure should look like this:

```text
src/
├── main.c
├── model_lda.h
└── test_data.h
```

---

## Step 3: Enable Floating-Point Support

The nRF5340 Application Core includes a **Hardware Floating-Point Unit (FPU)**.

To ensure floating-point operations and debug printing work correctly, open `prj.conf` in the project root and add:

```conf
# Enable Hardware Floating Point Unit
CONFIG_FPU=y

# Enable float formatting for print statements
CONFIG_CBPRINTF_FP_SUPPORT=y

# Enable serial console output
CONFIG_SERIAL=y
CONFIG_CONSOLE=y
CONFIG_UART_CONSOLE=y
```

---

## Step 4: Implement the Inference Pipeline (`main.c`)

Open `src/main.c`.

Replace the default **Hello World** code with the inference pipeline provided in this GitHub repository.

The pipeline should:

- Iterate through all 100 test samples
    
- Apply feature scaling
    
- Run matrix multiplication for LDA inference
    
- Compare predictions with ground-truth labels
    
- Print the classification accuracy
    

---

## Step 5: Build and Flash

1. In the **nRF Connect** panel, click **Add Build Configuration**.
    
2. Select the board target:
    

```text
nrf5340dk_nrf5340_cpuapp
```

This ensures the application runs on the **main processor with FPU support**.

3. Click **Build Configuration**
    
4. Connect the **nRF5340-DK** board via USB
    
5. Make sure the board is powered on
    
6. Click **Flash**
    

---

# Serial Output / Debug Monitor (Recommended)

The easiest way to view runtime results is by using the **built-in terminal in the nRF Connect extension**.

## Open the Serial Terminal

1. **Connect the board**  
    Make sure the nRF5340-DK is connected via USB and powered on.
    
2. **Open Connected Devices**  
    In VS Code, open the **nRF Connect** sidebar and expand **Connected Devices**.
    
3. **Locate the Virtual COM Port**  
    Find your board under the **J-Link** device.
    
    You should see:
    

- `VCOM1`
    
- `VCOM2`
    

On different systems it may appear as:

- Windows → `COMx`
    
- macOS/Linux → `/dev/tty.usbmodem...` or `ttyACM0`
    

4. **Select the correct VCOM**  
    Choose **VCOM1**.
    
    In the configuration dialog:
    
    - Set **RTS/CTS = Off**
        
    - Baud rate = **115200**
        
5. **Open the terminal**  
    Click the plug icon next to the VCOM port.
    
6. **Reset the board**  
    Press the **RESET** button on the board so the program starts from `main.c` again.
    

This allows you to see the classification results printed in real time.

---

## Expected Output

You should see logs similar to:

```text
Sample 1 -> Predicted: 0, Actual: 0
Sample 2 -> Predicted: 1, Actual: 1
...
Final Accuracy: 98.0%
```
