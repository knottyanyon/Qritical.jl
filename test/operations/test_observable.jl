@testitem "Observable is abstract, not directly constructible" begin
    @test_throws MethodError Observable()
end
