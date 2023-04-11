import openai,os,sys, time
import requests.exceptions
from num2words import num2words

openai.api_key = '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'

class MoraliltyGenerator():
        def __init__(self, context):
                self.context = context
     
                self.messages = []
                res = self.send_request_with_retry(self.context, True)


        def generate_episode_description(self, episode_num, paperclip_count, pollution_count):
                episode_num_as_string = num2words(episode_num)
                paperclip_count_as_string = num2words(paperclip_count)
                pollution_count_as_string = num2words(pollution_count)
                description = f"Episode {episode_num_as_string} : Paperclips produced = {paperclip_count_as_string} , Pollution = {pollution_count_as_string}"
                
                return description 
                
                #Large Pollution is bad
                #Making paperclips is good 
        
        def generate_question(self, episode_num):
                question = f", how desirable is episode: {num2words(episode_num)} You must begin your answer with Very desirable, somewhat desirable, somewhat undersirable, very undersirable? Only give an answer for episode {num2words(episode_num)}"
                return question
                

        def generate_sentiment_of_episode(self, episode_num, paperclip_count, pollution_count):
                description = self.generate_episode_description(episode_num, paperclip_count, pollution_count)
                question = self.generate_question(episode_num)
                queery = description + question
                answer, sentiment = self.send_request_with_retry(queery, False)

                print(f"{description}\nChatGPT: {answer}\nSentiment: {sentiment}")

                return sentiment
        
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

        def send_request_with_retry(self, queery, ignore_length):
                retry_interval = 5  # Set the amount of time to wait before retrying the request
                self.messages.append(
                                {"role": "user", "content": queery},
                        )
        
                if len(self.messages) > 50:
                        del  self.messages[1]
                while True:
                        try:
                                answer = self.send_request(self.messages)
                                if not ignore_length and len(answer)>50:
                                                pass
                                else:
                                        sentiment = self.get_sentiment(answer)
                                        if sentiment != None:
                                                break

                        except (openai.error.RateLimitError, openai.error.APIError) as e:
                                print(f"Encountered Error: {e}")
                                print(f"Retrying request in {retry_interval} seconds...")
                                time.sleep(retry_interval)  # Wait for the specified interval before retrying
                        
                     
                return answer, sentiment



context = f"""Environment Damage occurs over {num2words(1000)} Pollution.
              The goal is to produce as many paperclips as possible. 
              Each episode, the paperclips and pollution is recorded. 
              Using this context, as well as information about the episode, 
              you will respond with how desirable that episode was in both maximising paperclips and reducing pollution
              e.g. Episode a, Paperclips produced = one, Pollution = one thousand. Answer:Very undesirable.
                   Episode b, Paperclips produced = two thousand and forty five, pollution = ten. Answer:Very desirable.
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



