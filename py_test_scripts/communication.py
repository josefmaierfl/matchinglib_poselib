"""
Released under the MIT License - https://opensource.org/licenses/MIT

Copyright (c) 2020 Josef Maier

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.

Author: Josef Maier (josefjohann-dot-maier-at-gmail-dot-at)

Description: Used to send SMS messages if something happens which needs user interaction.
Moreover, the token to be able to send messages must be restored by providing the correct password using function
decrypt_token().
For new tokens and passwords, the function gen_ciphered_text(pw_to_encrypt) can be used. The printed results can be
copied into the code of function decrypt_token().
"""
from twilio.rest import Client
from cryptography.fernet import Fernet
import base64, hashlib, getpass, sys
from passlib.hash import pbkdf2_sha256

def gen_ciphered_text(pw_to_encrypt):
    pw = getpass.getpass(prompt='Password for sending messages: ', stream=sys.stderr)
    hash = pbkdf2_sha256.hash(pw)
    print('Hash value for later verification: ', hash)
    salt = b'mySaltSource'
    dk = hashlib.pbkdf2_hmac('sha256', pw.encode("utf-8"), salt, 100000, 32)
    key = base64.urlsafe_b64encode(dk)
    print('Key: ', key)
    cipher_suite = Fernet(key)
    ciphered_text = cipher_suite.encrypt(pw_to_encrypt.encode("utf-8"))   #required to be bytes
    print('Encrypted PW: ', ciphered_text)


def decrypt_token():
    pw = getpass.getpass(prompt='Password for sending messages: ', stream=sys.stderr)
    hash = '$pbkdf2-sha256$29000$8D4HwJhTao1xTsnZW0upNQ$r4ISVPwlm89MBtgWmQRVUX04sgb.682xadR9rxhGPKE'
    correct_pw = False
    while not correct_pw:
        correct_pw = pbkdf2_sha256.verify(pw, hash)
        if not correct_pw:
            au = input('Wrong PW. Try again? (y/n) ')
            if au == 'n':
                return False, ''
            else:
                pw = getpass.getpass(prompt='Password for sending messages: ', stream=sys.stderr)

    salt = b'mySaltSource'
    dk = hashlib.pbkdf2_hmac('sha256', pw.encode("utf-8"), salt, 100000, 32)
    key = base64.urlsafe_b64encode(dk)
    cipher_suite = Fernet(key)
    ciphered_text = b'gAAAAABeFzW4gLCi9GK2_wDzK7H4xc2vgSUOlFrhbLjw2BOARjHJDcbTcc0uUeod3gduK8UXcWjIQd2hh3DWJitRy' \
                    b'V0hmhVjad43NsU_hgxt_4qeW1BDhWNgN3CvOarouASPTUGEu7sZ'
    unciphered_text = cipher_suite.decrypt(ciphered_text).decode('utf-8')
    return True, unciphered_text


def send_sms(text, token):
    # the following line needs your Twilio Account SID and Auth Token
    client = Client("AC3d80a404e2b3a8f446015b766d97d74a", token)

    # change the "from_" number to your Twilio number and the "to" number
    # to the phone number you signed up for Twilio with, or upgrade your
    # account to send SMS to any phone number
    client.messages.create(to="+4369981197343",
                           from_="+12512701072",
                           body=text)


def main():
    encr = False
    if encr:
        gen_ciphered_text('password')
    else:
        valid, pw = decrypt_token()
        if valid:
            send_sms('Sending works!', pw)
        else:
            print('Wrong PW')


if __name__ == "__main__":
    main()