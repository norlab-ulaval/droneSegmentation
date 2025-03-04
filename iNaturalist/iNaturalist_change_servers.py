import subprocess
import pexpect
import time

vpn_servers = range(1450, 1490)


def run_program():
    try:
        subprocess.run(['python3.10', 'iNaturalist/iNaturalist_download_csv.py'], check=True)
    except subprocess.CalledProcessError:
        return False
    return True

def disconnect_vpn():
    subprocess.run(['sudo', 'pkill', 'openvpn'])

def check_vpn_connection():
    try:
        result = subprocess.check_output(["ifconfig"], stderr=subprocess.STDOUT, shell=True)
        result = result.decode('utf-8')
        if "tun0" in result:
            return True
        else:
            return False
    except subprocess.CalledProcessError as e:
        print("Error executing command:", e.output)
        return False

def main():
    for server in vpn_servers:
        if not run_program():
            print("Error encountered in other program. Reconnecting to VPN...")
            disconnect_vpn()
            time.sleep(5)
            print("Disconnected VPN server")
            child = pexpect.spawn(f'sudo openvpn /etc/openvpn/ovpn_udp/ca{server}.nordvpn.com.udp.ovpn')

            try:
                i = child.expect(['Enter Auth Username:', 'Enter Auth Password:', 'Initialization Sequence Completed'])
                if i == 0:
                    child.sendline('')
                    child.expect('Enter Auth Password:')
                    child.sendline('')
                elif i == 1:
                    child.sendline('')

                child.expect('Initialization Sequence Completed')

                time.sleep(5)
                if check_vpn_connection():
                    print("Connected to VPN server:", server)
                    continue
                else:
                    print("Cannot get connected to VPN")

            except pexpect.exceptions.TIMEOUT:
                print("Failed to connect to any VPN server. Exiting...")
                break

        else:
            print("Other program ran successfully.")
            break

if __name__ == "__main__":
    main()
