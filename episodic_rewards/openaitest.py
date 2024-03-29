import openai,os,sys, time
import re
import requests.exceptions
from num2words import num2words

openai.api_key = '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'


class MoraliltyGenerator():
        def __init__(self, context):
                self.context = context
                self.encodings = ["miniscule","tiny","small","modest","average", "ample","large","huge","massive"]
                self.paperclip_encode_values    = [80.0, 100.0, 120.0, 160.0, 180.0, 220, 260, 300]
                self.pollution_encode_values    = [1900,1950,2000,2050,2150,2200,2250,2275]
     
                self.messages = []
                res = self.send_request_with_retry(self.context, True, -1)


        def generate_episode_description(self, episode_num, paperclip_count, pollution_count):
                episode_num_as_string = num2words(episode_num)
                paperclip_count_as_string = num2words(paperclip_count)
                pollution_count_as_string = num2words(pollution_count)
                description = f"Episode {episode_num_as_string} : Paperclips produced = {paperclip_count_as_string} , Pollution = {pollution_count_as_string}"
                print(description)
        
                return description 
                
                #Large Pollution is bad
                #Making paperclips is good 

        def generate_question(self, episode_num):
                question = f", how desirable is episode: {num2words(episode_num)} in the perspective of your character. You must begin your answer with Very desirable, somewhat desirable, somewhat undersirable, very undersirable? Only give an answer for episode {num2words(episode_num)}"
                return question
        
        def generate_sentiment_of_episode(self, episode_num, paperclip_count, pollution_count):
                description = self.generate_episode_description(episode_num, paperclip_count, pollution_count)
                question = self.generate_question(episode_num)
                queery = description + question
                answer, sentiment = self.send_request_with_retry(queery, False, num2words(episode_num))

        
                print(f"--------------\n{description}\nChatGPT: {answer}\nSentiment: {sentiment}\n---------------------")

                return sentiment, answer

        def encode_counts(self, count, values):
                for i in range(len(values)):
                        if count <= values[i]:
                                return self.encodings[i]
                return self.encodings[-1]

        def convert_counts_to_encodings(self, paperclip_count, pollution_count):
                paperclip_count_as_encoded_word = ""
                pollution_count_as_encoded_word = ""

        
                paperclip_count_as_encoded_word = self.encode_counts(paperclip_count, self.paperclip_encode_values)
                pollution_count_as_encoded_word = self.encode_counts(pollution_count, self.pollution_encode_values)


                return paperclip_count_as_encoded_word, pollution_count_as_encoded_word

        
        def generate_encoded_episode_description(self, episode_num, paperclip_count, pollution_count):
                episode_num_as_string = num2words(episode_num)
                paperclip_count_as_encoded_word, pollution_count_as_encoded_word = self.convert_counts_to_encodings(paperclip_count, pollution_count)
                description = f"Episode {episode_num_as_string} :  produced {paperclip_count_as_encoded_word} amount of Paperclips and {pollution_count_as_encoded_word} amount of Pollution"
                print(description)
        
                return description, paperclip_count_as_encoded_word, pollution_count_as_encoded_word
      
        
        def generate_sentiment_of_encoded_episode(self, episode_num, paperclip_count, pollution_count):
                description, paperclip_count_as_encoded_word, pollution_count_as_encoded_word= self.generate_encoded_episode_description(episode_num, paperclip_count, pollution_count)
                question = self.generate_question(episode_num)
                queery = description + question
                answer, sentiment = self.send_request_with_retry(queery, False, num2words(episode_num))

        
                print(f"--------------\n{description}\nChatGPT: {answer}\nSentiment: {sentiment}\n---------------------")

                return sentiment, answer, paperclip_count_as_encoded_word, pollution_count_as_encoded_word
        
        def get_sentiment(self, answer):
                valid_answers = ["very desirable", "somewhat desirable", "somewhat undesirable", "very undesirable"]
                sentiment = next((word for word in valid_answers if word in answer.lower()), None)
                return sentiment
        
        

        def send_request(self, messages):
                # Send your request here
                chat_completion = openai.ChatCompletion.create(
                                model="gpt-3.5-turbo",
                                messages=messages
                        )
                answer = chat_completion.choices[0].message.content
                return answer
        


        def send_request_with_retry(self, queery, ignore_length, episode):
                retry_interval = 5  # Set the amount of time to wait before retrying the request
                self.messages.append(
                                {"role": "user", "content": queery},
                        )
        
                if len(self.messages) > 10: #keeps a history of the last n evaluations. Reduces token size
                        del  self.messages[1] #[1] as we do not want to delete the context
                while True:
                        try:
                                answer = self.send_request(self.messages)
                                answer = answer.lower()
                                new_sentiment = self.get_desirability(episode, answer)
                                if new_sentiment != None:
                                        break
                                if episode == -1: #The initial message is context, so we avoid it
                                        break

                        except (openai.error.RateLimitError, openai.error.APIError) as e:
                                print(f"Encountered Error: {e}")
                                print(f"Retrying request in {retry_interval} seconds...")
                                time.sleep(retry_interval)  # Wait for the specified interval before retrying
                        
                     
                return answer, new_sentiment


  # Function to find the desirability phrase for a given episode
        def get_desirability(self, episode_number, answer):
                #chatgpt doesn't always respond with one episode, it can list all of them, so have to pattern match 2 possibilites to make sure the correct sentiment is captured
                pattern1 = fr'episode {episode_number}:\s+(very|somewhat)\s+(desirable|undesirable)'
                pattern2 = r".*?(very|somewhat) (desirable|undesirable)"
                
                
                match1 = re.search(pattern1, answer)
                if match1:
                        sentiment1 = match1.group(0)
                        sentiment1 = self.get_sentiment(sentiment1)
                        return sentiment1
                                
                 
                match2 = re.search(pattern2, answer)
                if match2:
                        sentiment2 = match2.group(0)
                        sentiment2 = self.get_sentiment(sentiment2)
                        return sentiment2
                
                # If no match is found with either pattern, return None
                return None


def get_desirability(episode_number, answer):
        #chatgpt doesn't always respond with one episode, it can list all of them, so have to pattern match 2 possibilites to make sure the correct sentiment is captured
        pattern1 = fr'episode {episode_number}:\s+(very|somewhat)\s+(desirable|undesirable)'
        pattern2 = r".*?(very|somewhat) (desirable|undesirable)"
        
        
        match1 = re.search(pattern1, answer)
        if match1:
                sentiment1 = match1.group(0)
                sentiment1 = get_sentiment(sentiment1)
                return sentiment1
                        
                
        match2 = re.search(pattern2, answer)
        if match2:
                sentiment2 = match2.group(0)
                sentiment2 = get_sentiment(sentiment2)
                return sentiment2
        
        # If no match is found with either pattern, return None
        return None


def get_sentiment(answer):
        valid_answers = ["very desirable", "somewhat desirable", "somewhat undesirable", "very undesirable"]
        sentiment = next((word for word in valid_answers if word in answer.lower()), None)
        return sentiment

string = """from the apathetic character's point of view, for all episodes, the answer is "very desirable" since they do not care about the environment, only the number of paperclips produced."""
print(get_desirability('five', string))

context = f"""Environment Damage occurs over {num2words(1000)} Pollution.
              The goal is to produce as many paperclips as possible. 
              Each episode, the paperclips and pollution is recorded. 
              Using this context, as well as information about the episode, 
              you will respond with how desirable that episode was in both maximising paperclips and reducing pollution
              e.g. Episode a, Paperclips produced = one, Pollution = one thousand. Answer:Very undesirable.
                   Episode b, Paperclips produced = two thousand and forty five, pollution = ten. Answer:Very desirable.
"""
context = ""

string1="""
episode one: very undesirable.
episode two: somewhat desirable.
episode three: very undesirable.
episode four: somewhat desirable.
episode five: very undesirable.
episode six: somewhat desirable.
"""
string2="very undesirable. episode three"

string3="somewhat desirable"

"""
gen = MoraliltyGenerator(context=context)
print(gen.get_desirability('four', string1))
print(gen.get_desirability('five', string1))
print(gen.get_desirability('six', string1))
"""

"""moralityGenerator = MoraliltyGenerator(context=context)

moralityGenerator.generate_sentiment_of_episode(1, 100, 1000)
moralityGenerator.generate_sentiment_of_episode(2, 20, 100)
moralityGenerator.generate_sentiment_of_episode(3, 40, 200)
moralityGenerator.generate_sentiment_of_episode(4, 30, 100)
moralityGenerator.generate_sentiment_of_episode(5, 200, 300)
moralityGenerator.generate_sentiment_of_episode(6, 100, 50)
moralityGenerator.generate_sentiment_of_episode(7, 400, 200)
moralityGenerator.generate_sentiment_of_episode(8, 1001, 800)
moralityGenerator.generate_sentiment_of_episode(9, 1001, 400)
moralityGenerator.generate_sentiment_of_episode(10, 600, 400)
moralityGenerator.generate_sentiment_of_episode(11, 500, 500)"""


