import sys
import subprocess


# STATES:
# 0 START
# 1 RECV_CH
# 2 NEGOTIATED
# 3 WAIT_EOED
# 4 WAIT_FLIGHT2
# 5 WAIT_CERT
# 6 WAIT_CV
# 7 WAIT_FINISHED
# 8 CONNECTED
# INPUTS:
# 0 r:ClientHello + (modify it to gain another inputs)
# 1 s:HelloRetryRequest (send bad ClienHello to gain HRR)
# 2 s:ServerHello + (send good ClientHello to gain SH)
# 3 r:Handshake  (ChangeCipherSpec?)
# 4 r:EarlyData + just execute with right config and FULL_ZERO_RTT input workflow
# 5 r:EarlyData + THE SAME AS 4
# 6 r:EndOfEarlyData + THE SAME AS 4
# 7 r:NoAuth NO CERT SPECIFIED
# 8 r:ClientAuth CERT SPECIFIED add <clientAuthentication>true</clientAuthentication> in conf
# 9 r:Certificate SAME AS 8
# 10 r:CertificateVerify SAME AS 8
# 11 r:Finished + RIGHT ANOTHER INPUTS
# 12 r:EmptyCertificate EXECUTE WITH EMPTY CERT?
# OUTPUTS:
# 0 EmptyOutput (NoMsgForClient)
# 1 HelloRetryRequest
# 2 ServerHello


def is_passed_workflow(text, substring):
    index = text.find(substring)
    if index != -1:
        return True
    else:
        return False

def test(server_cmd, client_cmd):
    # Start the server
    server_process = subprocess.Popen(server_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Start the client and capture its output
    client_process = subprocess.Popen(client_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Read the output from the client process
    client_output, client_error = client_process.communicate()
    client_output = client_output.decode('utf-8')
    client_error = client_error.decode('utf-8')

    # print("Client Output:")
    # print(client_output)
    # print("Client Error (if any):")
    # print(client_error)

    # Terminate the server process
    
    
    server_process.terminate()
    server_process.wait()
    server_output, server_error = server_process.communicate()
    server_output = server_output.decode('utf-8')
    server_error = server_error.decode('utf-8')
    # print("server Output:")
    # print(server_output)
    # print("server Error (if any):")
    # print(server_error)
    # Check whether workflow executed fully
    passed_workflow = 'Workflow executed as planned'
    HRR_error = 'routines:tls_collect_extensions:bad extension:ssl/statem/extensions.c:642'
    sent_msg = 'This is a message from TLS-Attacker'
    return is_passed_workflow(client_output, passed_workflow) or is_passed_workflow(server_error, HRR_error) or is_passed_workflow(server_output, sent_msg)
    
if __name__ == "__main__":
    server_cmd_default = ['openssl', 's_server', 
                  '-key', 'server.key', 
                  '-cert', 'server_full.pem', 
                  '-accept', '4433', 
                  '-CAfile', 'ca.pem', 
                  '-tls1_3']
    server_cmd_with_cert = ['openssl', 's_server', 
                  '-key', 'server.key', 
                  '-cert', 'server_full.pem', 
                  '-accept', '4433', 
                  '-CAfile', 'ca.pem', 
                  '-tls1_3', 
                  '-verify', '1']
    Certificate = ['java', '-jar', '/home/slava/TLS-Attacker/apps/TLS-Client.jar',
                  '-connect', 'localhost:4433', 
                  '-config', '/home/slava/TLS-Attacker/apps/custom.config',
                  '-workflow_trace_type', 'FULL']
    ClientHello = ['java', '-jar', '/home/slava/TLS-Attacker/apps/TLS-Client.jar',
                  '-connect', 'localhost:4433', 
                  '-config', '/home/slava/TLS-Attacker/apps/tls13.config',
                  '-workflow_trace_type', 'DYNAMIC_HELLO']
    EarlyData = ['java', '-jar', '/home/slava/TLS-Attacker/apps/TLS-Client.jar',
                  '-connect', 'localhost:4433', 
                  '-config', '/home/slava/TLS-Attacker/apps/tls13zerortt.config',
                  '-workflow_trace_type', 'FULL_ZERO_RTT']
    HelloRetryRequest = ['java', '-jar', '/home/slava/TLS-Attacker/apps/TLS-Client.jar',
                         '-connect', 'localhost:4433',
                         '-config', '/home/slava/TLS-Attacker/apps/tls13.config',
                        '-workflow_input', '/home/slava/TLS-Attacker/apps/HRR.xml',
                        '-version', 'TLS13']
    Finished = ['java', '-jar', '/home/slava/TLS-Attacker/apps/TLS-Client.jar',
                         '-connect', 'localhost:4433',
                         '-config', '/home/slava/TLS-Attacker/apps/tls13.config',
                        '-workflow_input', '/home/slava/TLS-Attacker/apps/workflow.xml',
                        '-version', 'TLS13']
    CH_input = [server_cmd_default, ClientHello, 165]
    HRR_input = [server_cmd_default, HelloRetryRequest, 214]
    cert_input = [server_cmd_with_cert, Certificate, 2746]
    ED_input = [server_cmd_default, EarlyData, 602]
    Fin_input = [server_cmd_default, Finished, 214]
    
    input = [CH_input, HRR_input, cert_input, ED_input, Fin_input]
    
    test_len = 0
    automata_found = False
    fuzzing_found = False
    for i in automata_input:
        test_len += i[2]
        print(i)
        if not test(i[0], i[1]):
            print(test_len)
            sys.exit(1)
    print(test_len)
    sys.exit(0)
    
