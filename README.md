1. **Build the Docker Image**
   - Use the provided Dockerfile to build the Docker image.
   - Visual Studio Code will automatically detect the Dockerfile and prompt you to start the build.

2. **Open the Project in the Container**
   - Once the image is built, open the project inside the Docker container.
   - Visual Studio Code will detect the Dockerfile and prompt you to open the container.

3. **Build the Project with CMake**
   - When the project is running in the container, build it using CMake.
   - Visual Studio Code can automatically detect the CMakeLists.txt file and prompt you to build, or you can manually run the build command in the terminal.

4. **Locate the Bin Folder**
   - After the build completes, a **bin** folder will be created in your project directory, containing the compiled executables.

5. **Run the Infer Executable**
   - Open a terminal in the container.
   - Navigate to the **bin** folder and execute the following command:
     ```bash
     ./bin/infer path_to_test_data
     ```
7. Important Note:
   - If the audio is corupted or does not open we remove it from the dataset. so all the results after it will be wrong.
     if this a test case in the dataset and you expect us to put random results for it, please let us know. so we can fix it fast.