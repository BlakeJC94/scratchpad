module Tst
    include("Tmp.jl")
    import .Tmp
    #using .Tmp

    Tmp.say_hello()
    # say_hello()

    # your other test code here
end
