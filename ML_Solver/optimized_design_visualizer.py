import yaml

def calculateOptimizedValues(optimized_points: str):

    geometry = {

    }

    with open("geometry.yaml", "r") as f:
        data = yaml.safe_load(f)
    print(data)

if __name__ == "__main__":
    main()