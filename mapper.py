from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom

def prettify(elem):
    """Return a pretty-printed XML string for the Element."""
    rough_string = tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

class WorkflowBuilder:
    def __init__(self):
        self.workflowTrace = Element('workflowTrace')

    def add_transition(self, send_state, receive_state):
        send = SubElement(self.workflowTrace, 'Send')
        actionOptions = SubElement(send, 'actionOptions')
        messages = SubElement(send, 'messages')

        if send_state == 'CLIENT_HELLO':
            clientHello = SubElement(messages, 'ClientHello')
            extensions = SubElement(clientHello, 'extensions')
            SubElement(extensions, 'ECPointFormat')
            SubElement(extensions, 'EllipticCurves')
            SubElement(extensions, 'SignatureAndHashAlgorithmsExtension')
            SubElement(extensions, 'RenegotiationInfoExtension')

        elif send_state == 'SERVER_HELLO':
            serverHello = SubElement(messages, 'ServerHello')
            extensions = SubElement(serverHello, 'extensions')
            SubElement(extensions, 'ECPointFormat')
            SubElement(extensions, 'RenegotiationInfoExtension')

        elif send_state == 'CERTIFICATE':
            SubElement(messages, 'Certificate')

        elif send_state == 'SERVER_HELLO_DONE':
            SubElement(messages, 'ServerHelloDone')

        elif send_state == 'CLIENT_KEY_EXCHANGE':
            SubElement(messages, 'ECDHClientKeyExchange')

        elif send_state == 'CHANGE_CIPHER_SPEC':
            SubElement(messages, 'ChangeCipherSpec')

        elif send_state == 'FINISHED':
            SubElement(messages, 'Finished')

        receive = SubElement(self.workflowTrace, 'Receive')
        actionOptions = SubElement(receive, 'actionOptions')
        expectedMessages = SubElement(receive, 'expectedMessages')

        if receive_state == 'SERVER_HELLO':
            serverHello = SubElement(expectedMessages, 'ServerHello')
            extensions = SubElement(serverHello, 'extensions')
            SubElement(extensions, 'ECPointFormat')
            SubElement(extensions, 'RenegotiationInfoExtension')
            SubElement(serverHello, 'autoSetHelloRetryModeInKeyShare').text = 'true'
            SubElement(expectedMessages, 'Certificate')
            SubElement(expectedMessages, 'ECDHEServerKeyExchange')
            SubElement(expectedMessages, 'ServerHelloDone')

        elif receive_state == 'CLIENT_HELLO':
            clientHello = SubElement(expectedMessages, 'ClientHello')
            extensions = SubElement(clientHello, 'extensions')
            SubElement(extensions, 'ECPointFormat')
            SubElement(extensions, 'EllipticCurves')
            SubElement(extensions, 'SignatureAndHashAlgorithmsExtension')
            SubElement(extensions, 'RenegotiationInfoExtension')

        elif receive_state == 'CERTIFICATE':
            SubElement(expectedMessages, 'Certificate')

        elif receive_state == 'SERVER_HELLO_DONE':
            SubElement(expectedMessages, 'ServerHelloDone')

        elif receive_state == 'CLIENT_KEY_EXCHANGE':
            SubElement(expectedMessages, 'ECDHClientKeyExchange')

        elif receive_state == 'CHANGE_CIPHER_SPEC':
            SubElement(expectedMessages, 'ChangeCipherSpec')

        elif receive_state == 'FINISHED':
            SubElement(expectedMessages, 'Finished')

    def build(self):
        return prettify(self.workflowTrace)

builder = WorkflowBuilder()
builder.add_transition('CLIENT_HELLO', 'SERVER_HELLO')
builder.add_transition('SERVER_HELLO', 'CERTIFICATE')
builder.add_transition('CERTIFICATE', 'SERVER_HELLO_DONE')
builder.add_transition('SERVER_HELLO_DONE', 'CLIENT_KEY_EXCHANGE')
builder.add_transition('CLIENT_KEY_EXCHANGE', 'CHANGE_CIPHER_SPEC')
builder.add_transition('CHANGE_CIPHER_SPEC', 'FINISHED')
print(builder.build())