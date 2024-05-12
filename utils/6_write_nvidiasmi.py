import subprocess
import time

def write_memory_usage_to_file(file_path):
    while True:
        # Run nvidia-smi command and capture its output
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True)
            
            memory_usage = result.stdout.strip()
        except FileNotFoundError:
            print("nvidia-smi command not found. Make sure NVIDIA GPU drivers are installed.")
            return

        # Write memory usage to file
        with open(file_path, 'a') as file:
            datetime = time.strftime("%Y-%m-%d %H:%M:%S")
            file.write(f"{datetime}, {memory_usage}\n")
            print(f"{datetime} Memory usage {memory_usage}")

        time.sleep(1)  # Wait for 1 second before next iteration

if __name__ == "__main__":
    file_path = "gpu_memory_usage.txt"
    write_memory_usage_to_file(file_path)
