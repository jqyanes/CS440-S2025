import numpy as np
import random
from PIL import Image, ImageDraw
import heapq
import time


def generate_maze(rows, cols):
    maze = np.zeros((rows, cols))
    visited = []
    unvisited = []
    for i in range(0, rows):
        for j in range(0, cols):
            unvisited.append((i,j))
    stack = []
    
    # Select a random starting point
    start_row = random.randint(0, rows - 1)
    start_col = random.randint(0, cols - 1)
    maze[start_row, start_col] = 1
    visited.append((start_row, start_col))
    unvisited.remove((start_row, start_col))
    stack.append((start_row, start_col))
    
    while len(visited) < rows * cols:
        if stack:
            row,col = stack[-1]
            
        potential_neighbors = [(row-1,col),(row+1,col),(row,col-1),(row,col+1)]
      
        #Making sure neighbors are inbound
        neighbors = []
        for r, c in potential_neighbors:
            if 0 <= r < rows and 0 <= c < cols and (r, c) not in visited:
                neighbors.append((r,c))
        
        if neighbors:
            next_row, next_col = random.choice(neighbors)
            visited.append((next_row, next_col))
            unvisited.remove((next_row, next_col))
            # 1 is unblocked (0.7 chance), 0 is blocked (0.3 chance)
            if random.random() <= .7:
                maze[next_row, next_col] = 1
                stack.append((next_row, next_col))
            else:
                maze[next_row, next_col] = 0 
        else:
            if stack:
                stack.pop()
            else: 
                new_row,new_col =  random.choice(unvisited)
                maze[new_row, new_col] = 1
                visited.append((new_row, new_col))
                unvisited.remove((new_row, new_col))
                stack.append((new_row, new_col))
                
    return maze


def draw_maze_with_grid(maze, cell_size=20, filename="maze.png"):
    unblocked = []
    rows, cols = maze.shape
    img = Image.new("RGB", (cols * cell_size, rows * cell_size), color="white")
    draw = ImageDraw.Draw(img)

    for row in range(rows):
        for col in range(cols):
            #white is unblocked, black is blocked
            x0, y0 = col * cell_size, row * cell_size
            x1, y1 = x0 + cell_size, y0 + cell_size
            if maze[row, col] == 0:
                draw.rectangle([x0, y0, x1, y1], fill="black")
                # draw.text((x0+2,y0+2), ','.join((str(row),str(col))), fill="red")
            if maze[row, col] == 1:
                unblocked.append((row,col))
                # draw.text((x0+2,y0+2), ','.join((str(row),str(col))), fill="blue")
    for row in range(0, rows):
        y = row * cell_size
        draw.line([(0, y), (cols * cell_size, y)], fill="gray", width=1)
    for col in range(0, cols):
        x = col * cell_size
        draw.line([(x, 0), (x, rows * cell_size)], fill="gray", width=1)
    draw.line([(cols*cell_size,0),(cols*cell_size, rows*cell_size)], fill="gray", width=4)
    draw.line([(0,rows*cell_size),(cols*cell_size, rows*cell_size)], fill="gray", width=4)

    start = random.choice(unblocked)
    temp = unblocked.copy()
    temp.remove(start)
    end = random.choice(temp)
    
    #start green
    sx0, sy0 = start[1] * cell_size, start[0] * cell_size 
    sx1, sy1 = sx0 + cell_size, sy0 + cell_size
    draw.rectangle([sx0, sy0, sx1, sy1], fill="green", outline="black")

    #End red
    ex0, ey0 = end[1] * cell_size, end[0] * cell_size 
    ex1, ey1 = ex0 + cell_size, ey0 + cell_size
    draw.rectangle([ex0, ey0, ex1, ey1], fill="red", outline="black")

    img.save(filename)
    return start, end


def manhattan_distance(start, end):
    return abs(start[0] - end[0]) + abs(start[1] - end[1])


def A_Forward_Small_g_value(grid, start, goal):
    open_list = []
    closed_list = []
    heapq.heappush(open_list, (0 + manhattan_distance(start, goal), 0, start))
    
    previous_position = {}
    gValues = {start: 0}
    
    while open_list:
        i, g, current_position = heapq.heappop(open_list)
        
        if current_position == goal:
            return makePath(previous_position, current_position), closed_list
        
        closed_list.append(current_position)
        
        for row, col in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            neighbor = (current_position[0] + row, current_position[1] + col)
            
            if not (0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0])):
                continue
            
            if grid[neighbor[0]][neighbor[1]] == 0:
                continue
            
            if neighbor in closed_list:
                continue
            
            ong = g + 1
            if neighbor not in gValues or ong < gValues[neighbor]:
                gValues[neighbor] = ong
                f = ong + manhattan_distance(neighbor, goal)
                heapq.heappush(open_list, (f, ong, neighbor))
                previous_position[neighbor] = current_position
    
    return "No path found", closed_list


def A_Forward_Large_g_value(grid, start, goal, c):
    open_list = []
    closed_list = []
    heapq.heappush(open_list, (0 + manhattan_distance(start, goal), 0, start))
    
    previous_position = {}
    gValues = {start: 0}
    
    while open_list:
        i, g, current_position = heapq.heappop(open_list)
        
        if current_position == goal:
            return makePath(previous_position, current_position), closed_list
        
        closed_list.append(current_position)
        
        for row, col in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            neighbor = (current_position[0] + row, current_position[1] + col)
            
            if not (0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0])):
                continue
            
            if grid[neighbor[0]][neighbor[1]] == 0:
                continue
            
            if neighbor in closed_list:
                continue
            
            ong = g + 1
            if neighbor not in gValues or ong < gValues[neighbor]:
                gValues[neighbor] = ong
                f = c * (ong + manhattan_distance(neighbor, goal)) - ong
                heapq.heappush(open_list, (f, ong, neighbor))
                previous_position[neighbor] = current_position
    
    return "No path found", closed_list


def A_Backward_Large_g_value(grid, start, goal, c):
    open_list = []
    closed_list = []
    heapq.heappush(open_list, (0 + manhattan_distance(goal, start), 0, goal))
    
    previous_position = {}
    gValues = {goal: 0}
    
    while open_list:
        i, g, current_position = heapq.heappop(open_list)
        
        if current_position in closed_list:
            continue
                
        if current_position == start:
            return makePath(previous_position, current_position), closed_list
        
        closed_list.append(current_position)
        
        for row, col in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            neighbor = (current_position[0] + row, current_position[1] + col)
            
            if not (0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0])):
                continue
            
            if grid[neighbor[0]][neighbor[1]] == 0:
                continue
            
            ong = g + 1
            if neighbor not in gValues or ong < gValues[neighbor]:
                gValues[neighbor] = ong
                f = c * (ong + manhattan_distance(start, neighbor)) - ong
                heapq.heappush(open_list, (f, ong, neighbor))
                previous_position[neighbor] = current_position
    
    return "No path found", closed_list


def adaptive_a_star(grid, start, goal):
    open_list = []
    closed_list = []
    heapq.heappush(open_list, (manhattan_distance(start, goal), 0, start))

    previous_position = {}
    gValues = {start: 0}
    hValues = {}

    while open_list:
        f, g, current_position = heapq.heappop(open_list)

        if current_position in closed_list:
            continue

        if current_position == goal:
            goal_distance = gValues[current_position]
            for state in closed_list:
                hValues[state] = max(hValues.get(state, 0), goal_distance - gValues[state])
            return makePath(previous_position, current_position), closed_list
        
        closed_list.append(current_position)

        for row, col in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            neighbor = (current_position[0] + row, current_position[1] + col)

            if (neighbor in closed_list or 
                not (0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0])) or 
                grid[neighbor[0]][neighbor[1]] == 0):
                continue

            ong = g + 1
            if neighbor not in gValues or ong < gValues[neighbor]:
                gValues[neighbor] = ong
                h = hValues.get(neighbor, manhattan_distance(neighbor, goal))
                heapq.heappush(open_list, (ong + h, -ong, neighbor))
                previous_position[neighbor] = current_position

    return "No path found", closed_list



def makePath(previous_position, current_position):
    path = []
    while current_position in previous_position:
        path.append(current_position)
        current_position = previous_position[current_position]
    path.reverse()
    return path


rows = 101
cols = 101
c = rows+cols  

for i in range(50):
    maze = generate_maze(rows, cols)
    start,end = draw_maze_with_grid(maze, cell_size=20, filename=f"maze{i}.png")
    print(f"start: {start} end: {end}")

    
    start_time = time.time()
    path_smaller_g, closed_list = A_Forward_Small_g_value(maze, start, end)
    end_time = time.time()
    print("Smaller g-value forward path:", path_smaller_g)
    print(len(closed_list), "cells expanded:", closed_list)
    print("Smaller g-value forward runtime:", end_time - start_time, "seconds")

    start_time = time.time()
    path_larger_g, closed_list = A_Forward_Large_g_value(maze, start, end, c)
    end_time = time.time()
    print("Larger g-values forward path:", path_larger_g)
    print(len(closed_list), "cells expanded:", closed_list)
    print("Larger g-value forward runtime :", end_time - start_time, "seconds")
    
    start_time = time.time()
    path_backwards, closed_list = A_Backward_Large_g_value(maze, start, end, c)
    end_time = time.time()
    print("Larger g-values backward path:", path_backwards)
    print(len(closed_list), "cells expanded:", closed_list)
    print("Larger g-value backward runtime:", end_time - start_time, "seconds")
    
    start_time = time.time()
    path_adaptive, closed_list = adaptive_a_star(maze, start, end)
    end_time = time.time()
    print("Path with Adaptive A* algorithm:", path_adaptive)
    print(len(closed_list), "cells expanded:", closed_list)
    print("Runtime for Adaptive A* algorithm:", end_time - start_time, "seconds")
    
    
    print("-------------")
    
    # maze_test = np.array([[1,1,1,1,1], [1,1,0,1,1], [1,1,0,0,1], [1,1,0,0,1], [1,1,1,0,1]])
    # maze_test = np.array([[1,1,0,1,1,1,1,1,1,0], [1,1,1,1,0,1,1,0,1,1], [1,0,1,1,0,1,1,0,0,1], [1,1,1,1,1,1,1,0,0,1], [0,1,0,1,0,1,0,1,1,0], [0,1,1,1,1,1,0,1,1,1], [1,1,1,1,1,1,1,0,1,1], [0,1,1,1,0,1,1,1,0,1], [1,0,0,0,1,0,1,0,1,1], [1,1,1,0,1,1,1,1,1,1]])
    # unblocked_cells = draw_maze_with_grid(maze_test, cell_size=20, filename=f"maze_test.png")
    # start_test = (9,5)
    # end_test = (4,7)
    # print(f"start: {start_test} end: {end_test}")
    
    # # start_time = time.time()
    # # path_smaller_g, closed_list = A_Forward_Small_g_value(maze_test, start_test, end_test)
    # # end_time = time.time()
    # # print("Smaller g-value forward path:", path_smaller_g)
    # # print(len(closed_list), "cells expanded:", closed_list)
    # # print("Smaller g-value forward runtime:", end_time - start_time, "seconds")

    # start_time = time.time()
    # path_larger_g, closed_list = A_Forward_Large_g_value(maze_test, start_test, end_test, c)
    # end_time = time.time()
    # print("Larger g-values forward path:", path_larger_g)
    # print(len(closed_list), "cells expanded:", closed_list)
    # print("Larger g-value forward runtime :", end_time - start_time, "seconds")
    
    # start_time = time.time()
    # path, closed_list = A_Backward_Large_g_value(maze_test, start_test, end_test, c)
    # end_time = time.time()
    # print("Larger g-values backward path:", path)
    # print(len(closed_list), "cells expanded:", closed_list)
    # print("Larger g-value backward runtime:", end_time - start_time, "seconds")
    
    # print("-------------")
