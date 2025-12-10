# ssl_test.py
import ssl
import certifi

print("SSL Version:", ssl.OPENSSL_VERSION)
print("Certifi location:", certifi.where())
print("\nTrying to load certificates...")

try:
    context = ssl.create_default_context(cafile=certifi.where())
    print("✓ Certificates loaded successfully")
except Exception as e:
    print(f"❌ Error: {e}")