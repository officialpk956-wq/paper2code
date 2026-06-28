import requests

BASE_URL = "http://127.0.0.1:8000/api/auth"

def run_tests():
    print("Testing Auth Endpoints...")
    
    # 1. Register
    reg_data = {
        "email": "test_auth_check@example.com",
        "password": "Password123!",
        "name": "Test User"
    }
    r = requests.post(f"{BASE_URL}/register", json=reg_data)
    print(f"Register: {r.status_code}")
    if r.status_code not in (200, 201, 400): # 400 if already registered
        print(r.text)
        
    # 2. Login
    login_data = {
        "username": "test_auth_check@example.com",
        "password": "Password123!"
    }
    r = requests.post(f"{BASE_URL}/login", data=login_data)
    print(f"Login: {r.status_code}")
    
    access_token = None
    if r.status_code == 200:
        access_token = r.json().get("access_token")
        print("Login successful, got token.")
    
    # 3. Me
    if access_token:
        headers = {"Authorization": f"Bearer {access_token}"}
        r = requests.get(f"{BASE_URL}/me", headers=headers)
        print(f"Get Me: {r.status_code}")
        
    # 4. Forgot Password
    r = requests.post(f"{BASE_URL}/forgot-password", json={"email": "test_auth_check@example.com"})
    print(f"Forgot Password: {r.status_code}")
    
    print("All tests completed.")

if __name__ == "__main__":
    run_tests()
