import pytest

from fastapi.testclient import TestClient

from sqlalchemy.orm import Session


from backend.models import User

from backend.modules.auth.security.hashing import hash_password


_PASS = "Execute999!"



def _seed_user(db: Session, email: str) -> User:

    existing = db.query(User).filter_by(email=email).first()

    if existing:

        return existing

    u = User(

        email=email,

        name=email.split("@")[0],

        hashed_password=hash_password(_PASS),

        is_verified=True,

        is_email_verified=True,

    )

    db.add(u)

    db.commit()

    db.refresh(u)

    return u



def _login(client: TestClient, email: str) -> str:

    r = client.post("/api/auth/login", data={"username": email, "password": _PASS})

    assert r.status_code == 200, r.text

    return r.json()["access_token"]



def _auth(token: str) -> dict:

    return {"Authorization": f"Bearer {token}"}



class TestExecuteEndpoint:


    def test_runs_simple_python(self, client, db_session):

        user = _seed_user(db_session, "exec01@test.com")

        token = _login(client, user.email)


        r = client.post(

            "/api/dojo/execute",

            json={"code": 'print("hello world")', "stdin": ""},

            headers=_auth(token),

        )

        assert r.status_code == 200

        data = r.json()

        assert "stdout" in data

        assert "hello world" in data["stdout"]


    def test_all_tests_passed_marker(self, client, db_session):

        user = _seed_user(db_session, "exec02@test.com")

        token = _login(client, user.email)


        r = client.post(

            "/api/dojo/execute",

            json={"code": 'print("ALL_TESTS_PASSED")', "stdin": ""},

            headers=_auth(token),

        )

        assert r.status_code == 200

        assert "ALL_TESTS_PASSED" in r.json()["stdout"]


    def test_assertion_failure_visible_in_stdout(self, client, db_session):

        user = _seed_user(db_session, "exec03@test.com")

        token = _login(client, user.email)


        code = """

try:

    assert 1 == 2, "one is not two"

    print("ALL_TESTS_PASSED")

except AssertionError as e:

    print(f"ASSERTION_FAILED: {e}")

"""

        r = client.post(

            "/api/dojo/execute",

            json={"code": code, "stdin": ""},

            headers=_auth(token),

        )

        assert r.status_code == 200

        assert "ASSERTION_FAILED" in r.json()["stdout"]


    def test_requires_authentication(self, client, db_session):

        r = client.post(

            "/api/dojo/execute",

            json={"code": 'print("hi")', "stdin": ""},

        )

        assert r.status_code in (401, 403)


    def test_code_too_large_rejected(self, client, db_session):

        user = _seed_user(db_session, "exec05@test.com")

        token = _login(client, user.email)


        big_code = "🤖" * 20000  # 20k chars (passes Pydantic max_length=65536), 80k bytes (fails custom byte check)

        r = client.post(

            "/api/dojo/execute",

            json={"code": big_code, "stdin": ""},

            headers=_auth(token),

        )

        assert r.status_code == 400