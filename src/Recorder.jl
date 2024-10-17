module RecorderModule

using ..CoreModule: RecordType

"Assumes that `options` holds the user options::Options"
macro recorder(ex)
    return quote
        if $(esc(:options)).use_recorder
            $(esc(ex))
        end
    end
end

function find_iteration_from_record(key::String, record::RecordType)
    iteration = 0
    while haskey(record[key], "iteration$(iteration)")
        iteration += 1
    end
    return iteration - 1
end

end
