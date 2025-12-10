# network_test.py
import socket
import ssl

def test_telegram_connection():
    try:
        print("Testing DNS resolution...")
        ip = socket.gethostbyname("api.telegram.org")
        print(f"✓ DNS OK: {ip}")
        
        print("\nTesting TCP connection...")
        sock = socket.create_connection(("api.telegram.org", 443), timeout=10)
        print("✓ TCP connection OK")
        
        print("\nTesting SSL/TLS...")
        context = ssl.create_default_context()
        with context.wrap_socket(sock, server_hostname="api.telegram.org") as ssock:
            print(f"✓ SSL OK: {ssock.version()}")
            
        print("\n✅ All network tests passed!")
        return True
        
    except socket.gaierror as e:
        print(f"❌ DNS Error: {e}")
    except socket.timeout as e:
        print(f"❌ Connection Timeout: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
    return False

test_telegram_connection()