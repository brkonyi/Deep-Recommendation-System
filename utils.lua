--Returns the names of all files in a given directory using find
function getFileNames(directory)
    local i, t, popen = 0, {}, io.popen
    for filename in popen('find "'..directory..'" -maxdepth 1 -type f -printf "%f\n"'):lines() do
        i = i + 1
        t[i] = filename
    end
    return t
end

--Standard string split method
function split(string, delim)
    if delim == nil then
        delim = "%s"
    end
    local t = {}
    local i = 1
    for str in string.gmatch(string, "([^"..delim.."]+)") do
        t[i] = str
        i = i + 1
    end

    return t
end

--Shuffles a table/array of values
function shuffle(array, size)
    local rand = math.random
    assert(array, "shuffle() expected a table/array, got nil")
    local iterations = size
    local j

    for i = iterations, 2, -1 do
        j = rand(i)
        array[i], array[j] = array[j], array[i]
    end
end


