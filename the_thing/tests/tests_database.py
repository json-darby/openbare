import os
import tempfile
import unittest
import sqlite3

import database

TEMP_DATABASE_NAME = "heart_is_breaking_"


class BaseDatabaseTest(unittest.TestCase):
    def setUp(self):
        # TEMP FILE TEMP FILE TEMP FILE TEMP FILE
        self.db_fd, self.db_path = tempfile.mkstemp(prefix=TEMP_DATABASE_NAME, suffix=".db")
        os.close(self.db_fd)
        database.DATABASE_NAME = self.db_path
        database.create_table()

    def tearDown(self):
        try:
            os.remove(self.db_path)
        except OSError:
            pass

class TestRegisterUser(BaseDatabaseTest):
    def test_valid_registration(self):
        print("\nTEST: Valid registration for Will Smith")
        msg = database.register_user("WI123456S", "Will", "Smith", "password", "password")
        self.assertIn("successfully", msg.lower())

    def test_invalid_staff_id(self):
        print("\nTEST: Reject registration with malformed staff ID")
        msg = database.register_user("X1", "Emma", "Stone", "password", "password")
        self.assertIn("invalid staff id", msg.lower())

    def test_nonalpha_first_name(self):
        print("\nTEST: Reject registration when first name contains digits")
        msg = database.register_user("EM123456E", "Emm4", "Stone", "password", "password")
        self.assertIn("first name must contain only letters", msg.lower())

    def test_nonalpha_surname(self):
        print("\nTEST: Reject registration when surname contains digits")
        msg = database.register_user("EM123456E", "Emma", "St0ne", "password", "password")
        self.assertIn("surname must contain only letters", msg.lower())

    def test_passwords_do_not_match(self):
        print("\nTEST: Reject registration when passwords do not match")
        msg = database.register_user("EL123456M", "Elon", "Musk", "password", "test123")
        self.assertIn("passwords do not match", msg.lower())

    def test_password_too_short(self):
        print("\nTEST: Reject registration with password shorter than 5 chars")
        msg = database.register_user("EK123456R", "Keanu", "Reeves", "1234", "1234")
        self.assertIn("at least 5 characters", msg.lower())

    def test_duplicate_staff_id(self):
        print("\nTEST: Reject registration with duplicate staff ID")
        _ = database.register_user("JO123456N", "John", "Legend", "password", "password")
        msg = database.register_user("JO123456N", "John", "Doe", "password", "password")
        self.assertIn("already exists", msg.lower())

    def test_name_capitalisation(self):
        print("\nTEST: Automatically capitalise names on registration")
        _ = database.register_user("BR123456I", "brad", "pitt", "password", "password")
        conn = sqlite3.connect(self.db_path)
        fn, sn = conn.execute(
            "SELECT first_name, surname FROM users WHERE staff_id=?", 
            ("BR123456I",)
        ).fetchone()
        conn.close()
        self.assertEqual(fn, "Brad")
        self.assertEqual(sn, "Pitt")

    def test_min_length_password_edge(self):
        print("\nTEST: Accept registration with exactly 5-char password")
        msg = database.register_user("RI123456C", "Rick", "Astley", "test123", "test123")
        self.assertIn("successfully", msg.lower())

    def test_register_after_deletion(self):
        print("\nTEST: Allow re-registration after deletion of user")
        _ = database.register_user("TI123456N", "Taylor", "Swift", "password", "password")
        _ = database.delete_user("TI123456N")
        msg = database.register_user("TI123456N", "Taylor", "Swift", "password", "password")
        self.assertIn("successfully", msg.lower())

class TestLoginUser(BaseDatabaseTest):
    def setUp(self):
        super().setUp()
        # Register a valid user for login tests
        database.register_user("RI123456H", "Rihanna", "Haynes", "test123", "test123")

    def test_successful_login(self):
        print("\nTEST: Successful login for Rihanna Haynes")
        result = database.login_user("RI123456H", "test123")
        self.assertIsInstance(result, tuple)
        self.assertEqual(result[0].lower(), "login successful.")

    def test_invalid_staff_id_format(self):
        print("\nTEST: Reject login with malformed staff ID")
        msg = database.login_user("1234", "test123")
        self.assertIn("invalid staff id", msg.lower())

    def test_password_too_short(self):
        print("\nTEST: Reject login with password shorter than 5 characters")
        msg = database.login_user("RI123456H", "1234")
        self.assertIn("at least 5 characters", msg.lower())

    def test_nonexistent_staff_id(self):
        print("\nTEST: Reject login for nonexistent staff ID")
        msg = database.login_user("ZZ999999Z", "test123")
        self.assertIn("not found", msg.lower())

    def test_incorrect_password(self):
        print("\nTEST: Reject login with incorrect password")
        msg = database.login_user("RI123456H", "wrongpw")
        self.assertIn("incorrect password", msg.lower())

    def test_full_name_returned(self):
        print("\nTEST: Return correct full name on successful login")
        _, sid, fullname = database.login_user("RI123456H", "test123")
        self.assertEqual(sid, "RI123456H")
        self.assertEqual(fullname, "Rihanna Haynes")

    def test_case_sensitive_staff_id(self):  # Lower case, but stored as uppercase
        print("\nTEST: Allow login with lowercase staff ID, normalised to uppercase")
        result = database.login_user("ri123456h", "test123")
        self.assertIsInstance(result, tuple)
        self.assertEqual(result[0].lower(), "login successful.")

    def test_login_after_delete(self):
        print("\nTEST: Reject login after user deletion")
        _ = database.delete_user("RI123456H")
        msg = database.login_user("RI123456H", "test123")
        self.assertIn("not found", msg.lower())

    def test_multiple_logins(self):
        print("\nTEST: Allow multiple consecutive logins")
        r1 = database.login_user("RI123456H", "test123")
        r2 = database.login_user("RI123456H", "test123")
        self.assertEqual(r1, r2)

    def test_sql_error_handling(self):
        print("\nTEST: Handle database connection errors gracefully")
        database.DATABASE_NAME = os.getcwd()  # invalid path to force error
        msg = database.login_user("RI123456H", "test123")
        self.assertTrue("error" in msg.lower())


if __name__ == "__main__":
    print("\n*** DATABASE TESTS ***")
    print(". RegisterUser tests: 10")
    print(". LoginUser tests: 10")
    print(". Total tests : 20\n")
    unittest.main(verbosity=2)