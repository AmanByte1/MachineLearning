import random

class CarRacingGame:
    def __init__(self):
        self.your_car = 0
        self.ai_car = 0
        self.track_length = 100

    def play(self):
        print("Welcome to the Car Racing Game!")
        while True:
            print(f"You are at position {self.your_car} on the track.")
            print(f"AI is at position {self.ai_car}.")
            if self.your_car >= self.track_length or self.ai_car >= self.track_length:
                break
            move = input("Enter 'accelerate' to speed up, 'brake' to slow down, or 'exit' to quit: ")
            if move == 'accelerate':
                self.your_car += random.randint(1, 10)
                print(f"You accelerated by {random.randint(1, 10)} positions.")
            elif move == 'brake':
                self.your_car -= random.randint(-5, 5)
                print(f"You braked and moved back {random.randint(-5, 5)} positions.")
            elif move == 'exit':
                print("Thanks for playing! Game over.")
                break
            else:
                print("Invalid input. Try again.")
            if self.ai_car < self.track_length:
                self.ai_car += random.randint(1, 10)
                print(f"AI accelerated by {random.randint(1, 10)} positions.")

        if self.your_car >= self.track_length and self.ai_car >= self.track_length:
            print("It's a tie!")
        elif self.your_car >= self.track_length:
            print("You won! Congratulations!")
        else:
            print("AI won! Better luck next time!")

game = CarRacingGame()
game.play()
