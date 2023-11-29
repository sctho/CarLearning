# This Code is Heavily Inspired By The YouTuber: Cheesy AI && NeuralNine


import math
import random
import sys
import os

import neat
import pygame

# Constants
# WIDTH = 1600
# HEIGHT = 880

start_offset = 0

WIDTH = 1920
HEIGHT = 1080

CAR_SIZE_X = 60
CAR_SIZE_Y = 60

BORDER_COLOR = (255, 255, 255, 255)  #  To Crash on Hit

CAR_COLOR = (0, 209, 255, 255)
car_count = 1
car_index = 0

current_generation = 0  # Initialising Generation counter


class Car:
    def __init__(self, offset):
        # Load Car Sprite and Rotate
        self.sprite = pygame.image.load("car.png").convert()  # Convert Speeds Up A Lot
        self.sprite = pygame.transform.scale(
            self.sprite, (CAR_SIZE_X, CAR_SIZE_Y)
        )  # Scaling car size
        self.rotated_sprite = self.sprite

        # self.position = [690, 740] # Starting Position
        car_offset = (offset * -80)
        self.position = [890 + car_offset, 900]  # Starting Position
        self.angle = 0
        self.speed = 0

        print(self.position)

        self.speed_set = False  # Flag For Default Speed Later on

        self.center = [
            self.position[0] + CAR_SIZE_X / 2,
            self.position[1] + CAR_SIZE_Y / 2,
        ]  # Calculate Center

        self.radars = []  # List For Sensors / Radars
        self.drawing_radars = []  # Radars To Be Drawn

        self.alive = True  # Boolean To Check If Car is Crashed

        self.distance = 0  # Distance Driven
        self.time = 0  # Time Passed

    def draw(self, screen):
        screen.blit(self.rotated_sprite, self.position)  # Draw Sprite
        self.draw_radar(screen)  # OPTIONAL FOR SENSORS

    def draw_radar(self, screen):
        # Optionally Draw All Sensors / Radars
        for radar in self.radars:
            position = radar[0]
            pygame.draw.line(screen, (255, 0, 0), self.center, position, 1)
            pygame.draw.circle(screen, (255, 255, 0), position, 5)

    def check_collision(self, game_map, other_cars):
        self.alive = True  # Flag for car initially alive
        for point in self.corners:
            # If Any Corner Touches Border Color -> Crash
            # Assumes Rectangle
            if game_map.get_at((int(point[0]), int(point[1]))) == BORDER_COLOR:
                self.alive = False  # Collision occured
                break
        
        #for other_car in other_cars:
        #    if other_car != self and self.collides_with(other_car):
        #        self.alive = False  # Collision occurred with another car
        #        break

    def check_collision2(self, game_map, other_cars):
        self.alive = True  # Flag for car initially alive
        #for point in self.corners:
            # If Any Corner Touches Border Color -> Crash
            # Assumes Rectangle
         #   if game_map.get_at((int(point[0]), int(point[1]))) == BORDER_COLOR:
          #      self.alive = False  # Collision occured
           #     break
        
        for other_car in other_cars:
            if other_car != self and self.collides_with(other_car):
                self.alive = False  # Collision occurred with another car
                break
    
    def collides_with(self, other_car):
    # Simple collision check based on bounding boxes
        return (
            self.position[0] < other_car.position[0] + CAR_SIZE_X
            and self.position[0] + CAR_SIZE_X > other_car.position[0]
            and self.position[1] < other_car.position[1] + CAR_SIZE_Y
            and self.position[1] + CAR_SIZE_Y > other_car.position[1]
    )

    def check_radar(self, degree, game_map):
        length = 0
        x = int(
            self.center[0]
            + math.cos(math.radians(360 - (self.angle + degree))) * length
        )
        y = int(
            self.center[1]
            + math.sin(math.radians(360 - (self.angle + degree))) * length
        )

        # While We Don't Hit BORDER_COLOR AND length < 300 (just a max) -> go further and further
        while not game_map.get_at((x, y)) == BORDER_COLOR and length < 300:
            length = length + 1
            x = int(
                self.center[0]
                + math.cos(math.radians(360 - (self.angle + degree))) * length
            )
            y = int(
                self.center[1]
                + math.sin(math.radians(360 - (self.angle + degree))) * length
            )

        # Calculate Distance To Border And Append To Radars List
        dist = int(
            math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2))
        )
        self.radars.append([(x, y), dist])

    def update(self, game_map, other_cars):
        # Set The Speed To 20 For The First Time
        # Only When Having 4 Output Nodes With Speed Up and Down
        if not self.speed_set:
            self.speed = 20
            self.speed_set = True

        # Get Rotated Sprite And Move Into The Right X-Direction
        # Don't Let The Car Go Closer Than 20px To The Edge
        self.rotated_sprite = self.rotate_center(self.sprite, self.angle)
        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.position[0] = max(self.position[0], 20)
        self.position[0] = min(self.position[0], WIDTH - 120)

        # Increase Distance and Time
        self.distance += self.speed
        self.time += 1

        # Same For Y-Position
        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        self.position[1] = max(self.position[1], 20)
        self.position[1] = min(self.position[1], WIDTH - 120)

        # Calculate New Center
        self.center = [
            int(self.position[0]) + CAR_SIZE_X / 2,
            int(self.position[1]) + CAR_SIZE_Y / 2,
        ]

        # Calculate Four Corners
        # Length Is Half The Side
        length = 0.5 * CAR_SIZE_X
        left_top = [
            self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * length,
            self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * length,
        ]
        right_top = [
            self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * length,
            self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * length,
        ]
        left_bottom = [
            self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * length,
            self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * length,
        ]
        right_bottom = [
            self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * length,
            self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * length,
        ]
        self.corners = [left_top, right_top, left_bottom, right_bottom]

        # Check Collisions And Clear Radars
        self.check_collision(game_map, other_cars)
        self.radars.clear()

        # From -90 To 120 With Step-Size 45 Check Radar
        for d in range(-90, 120, 45):
            self.check_radar(d, game_map)

    def get_data(self):
        # Get Distances To Border
        radars = self.radars
        return_values = [0, 0, 0, 0, 0]
        for i, radar in enumerate(radars):
            return_values[i] = int(radar[1] / 30)

        return return_values
    
    def get_data2(self, other_cars):
    # Get Distances To Border and Other Cars
        radars = self.radars

    # Calculate the size of the return_values list based on the number of radars and other cars
        return_values = [0] * (5 + len(radars) + 2 * len(other_cars))

    # Populate values related to radars
        for i, radar in enumerate(radars):
            return_values[i] = int(radar[1] / 30)

    # Add information about other cars (distance and speed)
        for i, other_car in enumerate(other_cars):
         if other_car != self:
                distance_to_other_car = math.sqrt(
                    (other_car.position[0] - self.position[0]) ** 2
                    + (other_car.position[1] - self.position[1]) ** 2
                )
            # Indices for distance and speed are calculated dynamically
                return_values[len(radars) + i] = int(distance_to_other_car / 30)
                return_values[len(radars) + len(other_cars) + i] = int(other_car.speed / 5)

        return return_values[:9]

    def is_alive(self):
        # Basic Alive Function
        return self.alive

    def get_reward(self):
        # Calculate Reward (Maybe Change?)
        # return self.distance / 50.0
        return self.distance / (CAR_SIZE_X / 2)
    
    def get_reward2(self, other_cars):
        reward = 0

        # Check for collisions with other cars
        for other_car in other_cars:
            if other_car != self and self.collides_with(other_car):
                reward -= 100  # Apply a penalty for collisions
            else:
                reward += .1
        return reward

    def rotate_center(self, image, angle):
        # Rotate The Rectangle
        rectangle = image.get_rect()
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rectangle = rectangle.copy()
        rotated_rectangle.center = rotated_image.get_rect().center
        rotated_image = rotated_image.subsurface(rotated_rectangle).copy()
        return rotated_image

    def update2(self, game_map, other_cars):
        # Set The Speed To 20 For The First Time
        # Only When Having 4 Output Nodes With Speed Up and Down
        if not self.speed_set:
            self.speed = 20
            self.speed_set = True

        # Get Rotated Sprite And Move Into The Right X-Direction
        # Don't Let The Car Go Closer Than 20px To The Edge
        self.rotated_sprite = self.rotate_center(self.sprite, self.angle)
        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.position[0] = max(self.position[0], 20)
        self.position[0] = min(self.position[0], WIDTH - 120)

        # Increase Distance and Time
        self.distance += self.speed
        self.time += 1

        # Same For Y-Position
        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        self.position[1] = max(self.position[1], 20)
        self.position[1] = min(self.position[1], WIDTH - 120)

        # Calculate New Center
        self.center = [
            int(self.position[0]) + CAR_SIZE_X / 2,
            int(self.position[1]) + CAR_SIZE_Y / 2,
        ]

        # Calculate Four Corners
        # Length Is Half The Side
        length = 0.5 * CAR_SIZE_X
        left_top = [
            self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * length,
            self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * length,
        ]
        right_top = [
            self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * length,
            self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * length,
        ]
        left_bottom = [
            self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * length,
            self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * length,
        ]
        right_bottom = [
            self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * length,
            self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * length,
        ]
        self.corners = [left_top, right_top, left_bottom, right_bottom]

        # Check Collisions And Clear Radars
        self.check_collision2(game_map, other_cars)
        self.radars.clear()

        # From -90 To 120 With Step-Size 45 Check Radar
        for d in range(-90, 120, 45):
            self.check_radar(d, game_map)

def run_simulation(genomes, config):
    # Empty Collections For Nets and Cars
    nets = []
    cars = []

    # Initialize PyGame And The Display
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)

    # For All Genomes Passed Create A New Neural Network
    for i, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0
        cars.append(Car(start_offset))
        

    # Clock Settings
    # Font Settings & Loading Map
    clock = pygame.time.Clock()
    generation_font = pygame.font.SysFont("Arial", 30)
    alive_font = pygame.font.SysFont("Arial", 20)
    game_map = pygame.image.load("map2.png").convert()  # Convert Speeds Up A Lot

    global current_generation
    current_generation += 1

    # Simple Counter To Roughly Limit Time (Not Good Practice)
    counter = 0

    while True:
        # Exit On Quit Event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)

        # For Each Car Get The Acton It Takes
        for i, car in enumerate(cars):
            output = nets[i].activate(car.get_data())
            choice = output.index(max(output))
            if choice == 0:
                car.angle += 10  # Left
            elif choice == 1:
                car.angle -= 10  # Right
            elif choice == 2:
                if car.speed - 2 >= 12:
                    car.speed -= 2  # Slow Down
            else:
                car.speed += 2  # Speed Up

        # Check If Car Is Still Alive
        # Increase Fitness If Yes And Break Loop If Not
        still_alive = 0
        for i, car in enumerate(cars):
            if car.is_alive():
                still_alive += 1
                car.update(game_map, cars)
                genomes[i][1].fitness += car.get_reward()

        if still_alive == 0:
            break

        counter += 1
        if counter == 30 * 15:  # Stop After About 20 Seconds
            break

        # Draw Map And All Cars That Are Alive
        screen.blit(game_map, (0, 0))
        for car in cars:
            if car.is_alive():
                car.draw(screen)

        # Display Info
        text = generation_font.render(
            "Generation: " + str(current_generation), True, (0, 0, 0)
        )
        text_rect = text.get_rect()
        text_rect.center = (900, 450)
        screen.blit(text, text_rect)

        text = alive_font.render("Still Alive: " + str(still_alive), True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (900, 490)
        screen.blit(text, text_rect)

        pygame.display.flip()
        clock.tick(60)  # 60 FPS


def joint_run_simulation(genomes, config):
    # Empty Collections For Nets and Cars
    nets = []
    cars = []

    offsets = [0, 2]

    # Initialize PyGame And The Display
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)

    # For All Genomes Passed Create A New Neural Network
    for i, (genome_id, g) in enumerate(genomes):
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0

    # Limit the number of cars to 2
        if i < 2:
            adjusted_offset = i * 2  # You can adjust the offset calculation as needed
            cars.append(Car(adjusted_offset))
        else:
            break 
        

    # Clock Settings
    # Font Settings & Loading Map
    clock = pygame.time.Clock()
    generation_font = pygame.font.SysFont("Arial", 30)
    alive_font = pygame.font.SysFont("Arial", 20)
    game_map = pygame.image.load("map2.png").convert()  # Convert Speeds Up A Lot

    global current_generation
    current_generation += 1

    # Simple Counter To Roughly Limit Time (Not Good Practice)
    counter = 0

    while True:
        # Exit On Quit Event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)

        # For Each Car Get The Acton It Takes
        for i, car in enumerate(cars):
            output = nets[i].activate(car.get_data2(cars))
            choice = output.index(max(output))
            if choice == 0:
                car.angle += 10  # Left
            elif choice == 1:
                car.angle -= 10  # Right
            elif choice == 2:
                if car.speed - 2 >= 12:
                    car.speed -= 2  # Slow Down
            else:
                car.speed += 2  # Speed Up

        # Check If Car Is Still Alive
        # Increase Fitness If Yes And Break Loop If Not
        still_alive = 0
        for i, car in enumerate(cars):
            if car.is_alive():
                still_alive += 1
                car.update(game_map, cars)
                genomes[i][1].fitness += car.get_reward2(cars)

        if still_alive == 0:
            break

        counter += 1
        if counter == 30 * 30:  # Stop After About 20 Seconds
            break

        # Draw Map And All Cars That Are Alive
        screen.blit(game_map, (0, 0))
        for car in cars:
            if car.is_alive():
                car.draw(screen)

        # Display Info
        text = generation_font.render(
            "Generation: " + str(current_generation), True, (0, 0, 0)
        )
        text_rect = text.get_rect()
        text_rect.center = (900, 450)
        screen.blit(text, text_rect)

        text = alive_font.render("Still Alive: " + str(still_alive), True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (900, 490)
        screen.blit(text, text_rect)

        pygame.display.flip()
        clock.tick(60)  # 60 FPS


if __name__ == "__main__":
    # Load Config
    config_path = "./config.txt"
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    config2_path = "./config2.txt"
    config2 = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config2_path,
    )

    # Create Population And Add Reporters
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    population2 = neat.Population(config)
    population2.add_reporter(neat.StdOutReporter(True))
    stats2 = neat.StatisticsReporter()
    population2.add_reporter(stats2)

    #pe = neat.parallel.ParallelEvaluator(5, run_simulation)

    
    population.run(run_simulation, 5)

    best_genome1 = stats.best_genome()

    start_offset = 2
    population2.run(run_simulation, 5)

    best_genome2 = stats2.best_genome()

    joint_population = neat.Population(config2)
    joint_population.add_reporter(neat.StdOutReporter(True))
    stats3 = neat.StatisticsReporter()
    joint_population.add_reporter(stats3)

    
    if best_genome1:
        joint_population.population[1] = best_genome1

    if best_genome2:
        joint_population.population[2] = best_genome2

    # Run the joint simulation
    
    joint_population.run(joint_run_simulation, 10)
        #joint_run_simulation(list(joint_population.population.items()), config2)



    #joint_population.add_genome(0, best_genome1)
    #joint_population.add_genome(1, best_genome2)

    #joint_run_simulation(list(joint_population.population.items()), config)

    #joint_population.run(joint_run_simulation, 5)

