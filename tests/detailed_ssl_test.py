# detailed_ssl_test.py
import socket
import ssl
import certifi

def detailed_telegram_test():
    hostname = "api.telegram.org"
    port = 443
    
    try:
        print(f"Connecting to {hostname}:{port}...")
        sock = socket.create_connection((hostname, port), timeout=10)
        print("✓ TCP connection established")
        
        print("\nAttempting SSL handshake...")
        context = ssl.create_default_context(cafile=certifi.where())
        
        # Try with more verbose settings
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED
        
        # Wrap with timeout
        sock.settimeout(30)
        
        print("Wrapping socket with SSL...")
        ssock = context.wrap_socket(sock, server_hostname=hostname)
        
        print(f"✓ SSL handshake successful!")
        print(f"  Protocol: {ssock.version()}")
        print(f"  Cipher: {ssock.cipher()}")
        
        ssock.close()
        return True
        
    except socket.timeout as e:
        print(f"❌ Timeout during SSL handshake: {e}")
        print("\nThis suggests something is intercepting/blocking the SSL connection")
    except ssl.SSLError as e:
        print(f"❌ SSL Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    return False

detailed_telegram_test()

# Also check system-wide proxy settings
print("\n--- Checking for proxy settings ---")
import os
print("HTTP_PROXY:", os.environ.get('HTTP_PROXY', 'Not set'))
print("HTTPS_PROXY:", os.environ.get('HTTPS_PROXY', 'Not set'))
print("http_proxy:", os.environ.get('http_proxy', 'Not set'))
print("https_proxy:", os.environ.get('https_proxy', 'Not set'))