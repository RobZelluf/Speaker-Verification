speakers = open("SPEAKERS.TXT", 'r')
out = open("possible_speakers_clean.txt", 'w')
for i in range(12):
    next(speakers)
for line in speakers:
    cur = line.split("|")
    try:
        cur[3] = float(cur[3])
        cur[1] = cur[1][1:-1]
        if cur[1] == "F" and cur[3] >= 25 and "clean-100" in cur[2]:
            out.write(line)
    except ValueError:
        print("hui")