'''
Requirements:

pip install ibm_watson

'''


from ibm_watson import ToneAnalyzerV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

class toneanalyzer:
    def _init_(self,apikey,url,version):
        self.apikey=apikey
        self.url=url
    def analyze(self,chat):
       return self.ta.tone_chat(chat).get_result()
    def initialize(self,version):
        authenticator=IAMAuthenticator(self.apikey)
        self.ta=ToneAnalyzerV3(version=version,authenticator=authenticator)
        self.ta.set_service_url(self.url)
    def tone(self,chat):
        utterance_tones=[]
        dictionary=self.analyze(chat)
        for i in dictionary['utterances_tone']:
          utterance_tones+=i['tones']

tone=toneanalyzer()
tone.initialize(version)
print(tone(chat))