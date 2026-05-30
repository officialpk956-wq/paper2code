import os
import json
import time
import subprocess
import tempfile
import redis

# Redis connection (mock setup)
# redis_client = redis.Redis(host='localhost', port=6379, db=0)

def execute_submission(submission_id, code, test_cases):
    """
    Simulates picking up a code execution job and running it in an isolated Docker container.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        solution_path = os.path.join(tmpdir, "solution.py")
        
        # Append tests to the code
        full_code = code + "\n\n" + "#" * 20 + "\n# TESTS\n"
        for test in test_cases:
            # Simple assertions based on test JSON structure
            full_code += f"assert {test['expression']} == {test['expected']}, '{test.get('message', 'Test failed')}'\n"
            
        with open(solution_path, "w") as f:
            f.write(full_code)
            
        print(f"Running execution job for submission {submission_id} in {tmpdir}")
        
        # In a real environment, we would run:
        # docker_cmd = [
        #     "docker", "run", "--rm", 
        #     "--network", "none",
        #     "--memory", "256m",
        #     "--cpus", "0.5",
        #     "-v", f"{tmpdir}:/app",
        #     "python:3.10-slim",
        #     "python", "/app/solution.py"
        # ]
        
        # Simulated run (for local dev without docker daemon)
        start_time = time.time()
        try:
            # We just run the python file locally for mockup, with a timeout
            process = subprocess.run(
                ["python", solution_path],
                capture_output=True,
                text=True,
                timeout=2.0
            )
            duration = (time.time() - start_time) * 1000
            
            if process.returncode == 0:
                return {"status": "SUCCESS", "logs": process.stdout, "duration_ms": duration}
            else:
                return {"status": "FAILED", "logs": process.stderr, "duration_ms": duration}
                
        except subprocess.TimeoutExpired:
            return {"status": "TIMEOUT", "logs": "Execution timed out (2.0s limit)", "duration_ms": 2000.0}
        except Exception as e:
            return {"status": "ERROR", "logs": str(e), "duration_ms": 0}

def worker_loop():
    print("Worker started. Listening for tasks...")
    # while True:
    #     task = redis_client.blpop("execution_queue", timeout=0)
    #     if task:
    #         data = json.loads(task[1])
    #         result = execute_submission(data['submission_id'], data['code'], data['test_cases'])
    #         redis_client.rpush(f"submission_result_{data['submission_id']}", json.dumps(result))

if __name__ == "__main__":
    worker_loop()
