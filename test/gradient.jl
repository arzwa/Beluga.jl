@testset "Gradient (profile)" begin
    g = [0.0, 0.21654, -4.29457, 0.28282, -0.07077, -1.77586, -2.57598,
        6.25627, -2.82879, 1.81408, 3.84539, 19.34362, -0.47348, -1.39255,
        -2.83181, 27.41048, -1.44903, -2.66647, 3.29678, 0.0, -0.08539,
        2.6401, -0.37875, 0.12269, -1.05525, -1.48215, -1.38917, 7.49706,
        0.25451, -2.1514, -10.16407, 0.06406, 1.94643, 11.01334, -1.90087,
        1.26579, 22.01815, -3.40453, 49.2567]
    d, p = DLWGD(s4, df3, 2., 1., 0.9, Branch)
    g_ = Beluga.gradient(d, p)
    for i=1:length(g)
        @test isapprox(g[i], g_[i], atol=0.0001)
    end
end
