#!/usr/bin/env python
# __author__ = 'Eric Van Cleve'
# __copyright__ = 'Copyright 2019, Proofpoint Inc'
# __license__ = 'GPL v.3'
# __version__ = '3.0.1'
# __email__ = 'evancleve@proofpoint.com'
# __status__ = 'Production'


import re
import string
import sys
from argparse import ArgumentParser
from base64 import urlsafe_b64decode

if sys.version_info[0] < 3:
    from urllib import unquote
    import HTMLParser

    htmlparser = HTMLParser.HTMLParser()
    unescape = htmlparser.unescape
    from string import maketrans
else:
    from urllib.parse import unquote
    from html import unescape

    maketrans = str.maketrans


class URLDefenseDecoder(object):
    """src: https://files.mtstatic.com/site_6638/draft_177/2?Expires=1635089899&Signature=Hjf344YPmEcpCRoATv405dT28LsJFWvd402bZ8IB59v1WIBrvl5ralTZvlYnrtV3rvf22449MQZ9WTfQoVDgafU~N9ToKHFDRM-DzO-HM3AVBbKgPNPQAifCvqCsWdw423qnkIRp15XnRE--86ogXYfJ6eC78EdLn8zRWMsK9p8_&Key-Pair-Id=APKAJ5Y6AV4GI7A555NA"""

    @staticmethod
    def __init__():
        URLDefenseDecoder.ud_pattern = re.compile(r'https://urldefense(?:\.proofpoint)?\.com/(v[0-9])/')
        URLDefenseDecoder.v1_pattern = re.compile(r'u=(?P<url>.+?)&k=')
        URLDefenseDecoder.v2_pattern = re.compile(r'u=(?P<url>.+?)&[dc]=')
        URLDefenseDecoder.v3_pattern = re.compile(r'v3/__(?P<url>.+?)__;(?P<enc_bytes>.*?)!')
        URLDefenseDecoder.v3_token_pattern = re.compile(r"\*(\*.)?")
        URLDefenseDecoder.v3_run_mapping = {}
        run_values = string.ascii_uppercase + string.ascii_lowercase + string.digits + '-' + '_'
        run_length = 2
        for value in run_values:
            URLDefenseDecoder.v3_run_mapping[value] = run_length
            run_length += 1

    def decode(self, rewritten_url):
        match = self.ud_pattern.search(rewritten_url)
        if match:
            if match.group(1) == 'v1':
                return self.decode_v1(rewritten_url)
            elif match.group(1) == 'v2':
                return self.decode_v2(rewritten_url)
            elif match.group(1) == 'v3':
                return self.decode_v3(rewritten_url)
            else:
                raise ValueError('Unrecognized version in: ', rewritten_url)
        else:
            raise ValueError('Does not appear to be a URL Defense URL')

    def decode_v1(self, rewritten_url):
        match = self.v1_pattern.search(rewritten_url)
        if match:
            url_encoded_url = match.group('url')
            html_encoded_url = unquote(url_encoded_url)
            url = unescape(html_encoded_url)
            return url
        else:
            raise ValueError('Error parsing URL')

    def decode_v2(self, rewritten_url):
        match = self.v2_pattern.search(rewritten_url)
        if match:
            special_encoded_url = match.group('url')
            trans = maketrans('-_', '%/')
            url_encoded_url = special_encoded_url.translate(trans)
            html_encoded_url = unquote(url_encoded_url)
            url = unescape(html_encoded_url)
            return url
        else:
            raise ValueError('Error parsing URL')

    def decode_v3(self, rewritten_url):
        def replace_token(token):
            if token == '*':
                character = self.dec_bytes[self.current_marker]
                self.current_marker += 1
                return character
            if token.startswith('**'):
                run_length = self.v3_run_mapping[token[-1]]
                run = self.dec_bytes[self.current_marker:self.current_marker + run_length]
                self.current_marker += run_length
                return run

        def substitute_tokens(text, start_pos=0):
            match = self.v3_token_pattern.search(text, start_pos)
            if match:
                start = text[start_pos:match.start()]
                built_string = start
                token = text[match.start():match.end()]
                built_string += replace_token(token)
                built_string += substitute_tokens(text, match.end())
                return built_string
            else:
                return text[start_pos:len(text)]

        match = self.v3_pattern.search(rewritten_url)
        if match:
            url = match.group('url')
            encoded_url = unquote(url)
            enc_bytes = match.group('enc_bytes')
            enc_bytes += '=='
            self.dec_bytes = (urlsafe_b64decode(enc_bytes)).decode('utf-8')
            self.current_marker = 0
            return substitute_tokens(encoded_url)

        else:
            raise ValueError('Error parsing URL')


def main():
    parser = ArgumentParser(prog='URLDefenseDecode', description='Decode URLs rewritten by URL Defense. Supports v1, v2, and v3 URLs.')
    parser.add_argument('rewritten_urls', nargs='+')
    args = parser.parse_args()
    urldefense_decoder = URLDefenseDecoder()
    for rewritten_url in args.rewritten_urls:
        try:
            print(urldefense_decoder.decode(rewritten_url))
        except ValueError as e:
            print(e)


if __name__ == '__main__':
    main()
