import os
import shutil
import tempfile
import unittest
import random
import string

import pyzipper
import encryption

ENCRYPTION_FOLDER = "inference_packaged"

''' If you run this test while reviewing the code, THIS WILL DELETE ALL SAVED INFERENCES '''

class TestExtendedEncryption(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_folder = ENCRYPTION_FOLDER
        if os.path.isdir(cls.test_folder):
            shutil.rmtree(cls.test_folder)
        os.makedirs(cls.test_folder, exist_ok=True)

        cls.img_dir = tempfile.mkdtemp()
        cls.img1 = os.path.join(cls.img_dir, "dummy_image_1.png")
        cls.img2 = os.path.join(cls.img_dir, "dummy_image_2.png")
        with open(cls.img1, "wb") as f:
            f.write(os.urandom(32))
        with open(cls.img2, "wb") as f:
            f.write(os.urandom(64))

        cls.records = []
        for i in range(1000):
            staff_id = f"AB{str(i).zfill(6)}Z"
            password = "".join(random.choices(string.ascii_letters + string.digits, k=8))
            msg = encryption.encrypt_and_package_images(
                [cls.img1, cls.img2],
                staff_id,
                password,
                full_name=f"User{i}",
                image_source="Test",
                inference_time=0.01 * i
            )
            zip_path = msg.split(" - '")[-1].rstrip("'")
            cls.records.append((zip_path, f"{staff_id}{password}"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_folder, ignore_errors=True)
        shutil.rmtree(cls.img_dir, ignore_errors=True)

    def test_archive_count(self):
        files = [f for f in os.listdir(self.test_folder) if f.endswith(".zip")]
        self.assertEqual(len(files), 1000)

    def test_correct_password_opens(self):
        for zip_path, pwd in self.records:
            with self.subTest(zip=zip_path):
                with pyzipper.AESZipFile(zip_path, 'r') as zf:
                    zf.setpassword(pwd.encode())
                    names = zf.namelist()
                    self.assertIn(os.path.basename(self.img1), names)
                    self.assertIn(os.path.basename(self.img2), names)
                    self.assertIn("audit.txt", names)

    def test_wrong_password_fails(self):
        for zip_path, correct_pwd in self.records:
            wrong_pwd = "WRONG" + correct_pwd[::-1]
            with self.subTest(zip=zip_path):
                with self.assertRaises(RuntimeError):
                    with pyzipper.AESZipFile(zip_path, 'r') as zf:
                        zf.setpassword(wrong_pwd.encode())
                        _ = zf.read(os.path.basename(self.img1))

    def test_bruteforce_simulation(self):
        for zip_path, correct_pwd in self.records:
            candidates = [correct_pwd] + [
                "".join(random.choices(string.ascii_letters + string.digits, k=len(correct_pwd)))
                for _ in range(5)
            ]
            success = 0
            for pwd in candidates:
                try:
                    with pyzipper.AESZipFile(zip_path, 'r') as zf:
                        zf.setpassword(pwd.encode())
                        _ = zf.read("audit.txt")
                    success += 1
                except RuntimeError:
                    pass
            with self.subTest(zip=zip_path):
                self.assertEqual(success, 1)


if __name__ == "__main__":
    print("\n*** ENCRYPTION TESTS ***")
    print(". Archives generated: 1000")
    print(". Correct-password tests: 1000")
    print(". Wrong-password tests: 1000")
    print(". Brute-force sims: 1000 (each 6 attempts)\n")
    unittest.main(verbosity=2)
