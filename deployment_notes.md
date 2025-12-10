1. Log in to DigitalOcean and click Create -> Droplets.
2. Choose Region: Pick the one closest to you (e.g., New York, Amsterdam, San Francisco).
3. Choose Image: Select Ubuntu 24.04 (LTS).
4. Choose Size:
- Select Basic (Shared CPU).
- Select Regular (Disk type).
- The Pick: Select the $24/month option.
-- 4 GB RAM (Crucial for embeddings/AI workflows + Docker overhead).
-- 2 CPUs (Helps processing 4 pipelines at once).
-- 80 GB SSD.
Note: The $12/month (2GB RAM) might work, but it's risky with AI/Embeddings. The $24 option is the safe "set it and forget it" zone.
5. Authentication Method
- open terminal 
- run this: ssh-keygen -t ed25519 -C "theleioploomikros@protonmail.com"
Press Enter 3 times:
When it asks for a file location (Keep default).
When it asks for a passphrase (leave empty for now to keep it simple).
- copy key to the clipboard: pbcopy < ~/.ssh/id_ed25519.pub
- When creating the Droplet, look for the "Authentication" section.
- Select SSH Key.[2][3]
- Click "Add New SSH Key".
- Paste your key (cmd+v).[1]
- Name it "theleioplo".