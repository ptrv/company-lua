local function addmatch(word, kind, args, returns, doc)
    word = word and word or ""
    kind = kind and kind or ""
    args = args and args or ""
    returns = returns and returns or ""
    doc = doc and doc or ""
    print(string.format("word:%s,kind:%s,args:%s,returns:%s,doc:%s", word, kind, args, returns, doc))
end

local getPath = function(str,sep)
    sep=sep or'/'
    return str:match("(.*"..sep..")")
end

local thisDir = getPath(arg[0])
package.path = package.path .. ";" .. thisDir .. "?.lua"

function string.starts(String,Start)
    return string.sub(String,1,string.len(Start))==Start
end

local function getValueForKey(t, key)
    for k, v in pairs(t) do
        if k == key then
            if v.childs then
                return v.childs
            end
        end
    end
    return nil
end

local function generateList()
    local prefix = arg[1]
    local prefixTable = {}
    local count = 1
    if not prefix then
        return
    end

    local lastCharIsTrigger = string.sub(prefix, -1) == '.'

    for str in string.gmatch(prefix, "[^.]+") do
        prefixTable[count] = str
        count = count + 1
    end

    local status, module = pcall(require, "api/baselib")
    if status and module then

        local mod = module

        for i = 1, #prefixTable do
            if i == #prefixTable then
                if lastCharIsTrigger then
                    local val = getValueForKey(mod, prefixTable[i])
                    if val then
                        mod = val
                    else
                        break
                    end
                end
                for k, v in pairs(mod) do
                    if string.starts(k, prefixTable[i]) or lastCharIsTrigger then
                        local word, kind, args, returns, doc
                        word = k
                        kind = v.type and v.type
                        args = v.args and v.args
                        returns = v.returns and v.returns
                        doc = v.description and v.description
                        addmatch(word, kind, args, returns, doc)
                    end
                end
            else
                local val = getValueForKey(mod, prefixTable[i])
                if val and not cache[val] then
                    mod = val
                else
                    break
                end
            end
        end
    end
end

generateList()
