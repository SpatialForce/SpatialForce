//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

namespace vox::fields {
template<>
struct base_template_geometry_t<Tetrahedron> {
    using base = Tetrahedron;
    static constexpr uint32_t dim = Tetrahedron::dim;
    using point_t = Tetrahedron::point_t;

    CUDA_CALLABLE static constexpr auto points() {
        return fixed_array_t<point_t, 4>{
            point_t{0.0, 0.0, 0.0},
            point_t{1.0, 0.0, 0.0},
            point_t{0.0, 1.0, 0.0},
            point_t{0.0, 0.0, 1.0}};
    }

    CUDA_CALLABLE static constexpr auto geometry_dim0() {
        return fixed_array_t<static_geometry_t<1>, 4>{
            static_geometry_t<1>{
                .ind = 0,
                .vtx = {0},
                .bnd = {0},
            },
            static_geometry_t<1>{
                .ind = 1,
                .vtx = {1},
                .bnd = {1},
            },
            static_geometry_t<1>{
                .ind = 2,
                .vtx = {2},
                .bnd = {2},
            },
            static_geometry_t<1>{
                .ind = 3,
                .vtx = {3},
                .bnd = {3},
            }};
    }

    CUDA_CALLABLE static constexpr auto geometry_dim1() {
        return fixed_array_t<static_geometry_t<2>, 6>{
            static_geometry_t<2>{
                .ind = 0,
                .vtx = {0, 1},
                .bnd = {0, 1},
            },
            static_geometry_t<2>{
                .ind = 1,
                .vtx = {0, 2},
                .bnd = {0, 2},
            },
            static_geometry_t<2>{
                .ind = 2,
                .vtx = {0, 3},
                .bnd = {0, 3},
            },
            static_geometry_t<2>{
                .ind = 3,
                .vtx = {2, 3},
                .bnd = {2, 3},
            },
            static_geometry_t<2>{
                .ind = 4,
                .vtx = {1, 3},
                .bnd = {1, 3},
            },
            static_geometry_t<2>{
                .ind = 5,
                .vtx = {1, 2},
                .bnd = {1, 2},
            },
        };
    }

    CUDA_CALLABLE static constexpr auto geometry_dim2() {
        return fixed_array_t<static_geometry_t<3>, 4>{
            static_geometry_t<3>{
                .ind = 0,
                .vtx = {1, 2, 3},
                .bnd = {3, 4, 5},
            },
            static_geometry_t<3>{
                .ind = 1,
                .vtx = {0, 2, 3},
                .bnd = {3, 2, 1},
            },
            static_geometry_t<3>{
                .ind = 2,
                .vtx = {0, 1, 3},
                .bnd = {4, 2, 0},
            },
            static_geometry_t<3>{
                .ind = 3,
                .vtx = {0, 1, 2},
                .bnd = {5, 1, 0},
            },
        };
    }

    CUDA_CALLABLE static constexpr auto geometry_dim3() {
        return fixed_array_t<static_geometry_t<4>, 1>{
            static_geometry_t<4>{
                .ind = 0,
                .vtx = {0, 1, 2, 3},
                .bnd = {0, 1, 2, 3},
            },
        };
    }
};

template<>
struct template_geometry_t<Tetrahedron, 1> : public base_template_geometry_t<Tetrahedron> {
    CUDA_CALLABLE static constexpr auto quadrature_info() {
        return quadrature_info_t<dim, 1>{
            .alg_acc = 1,
            .pnts = {point_t{0.25000000, 0.25000000, 0.25000000}},
            .weights = {1.0}};
    }
};

template<>
struct template_geometry_t<Tetrahedron, 2> : public base_template_geometry_t<Tetrahedron> {
    CUDA_CALLABLE static constexpr auto quadrature_info() {
        return quadrature_info_t<dim, 4>{
            .alg_acc = 2,
            .pnts = {point_t{0.13819660, 0.13819660, 0.13819660},
                     point_t{0.58541020, 0.13819660, 0.13819660},
                     point_t{0.13819660, 0.58541020, 0.13819660},
                     point_t{0.13819660, 0.13819660, 0.58541020}},
            .weights = {0.25, 0.25, 0.25, 0.25}};
    }
};

template<>
struct template_geometry_t<Tetrahedron, 3> : public base_template_geometry_t<Tetrahedron> {
    CUDA_CALLABLE static constexpr auto quadrature_info() {
        return quadrature_info_t<dim, 5>{
            .alg_acc = 3,
            .pnts = {
                point_t{0.25000000, 0.25000000, 0.25000000},
                point_t{0.16666667, 0.16666667, 0.16666667},
                point_t{0.50000000, 0.16666667, 0.16666667},
                point_t{0.16666667, 0.50000000, 0.16666667},
                point_t{0.16666667, 0.16666667, 0.50000000},
            },
            .weights = {-0.8, 0.45, 0.45, 0.45, 0.45}};
    }
};

template<>
struct template_geometry_t<Tetrahedron, 4> : public base_template_geometry_t<Tetrahedron> {
    CUDA_CALLABLE static constexpr auto quadrature_info() {
        return quadrature_info_t<dim, 11>{
            .alg_acc = 4,
            .pnts = {
                point_t{0.2500000000000000, 0.2500000000000000, 0.2500000000000000},
                point_t{0.7857142857142857, 0.0714285714285714, 0.0714285714285714},
                point_t{0.0714285714285714, 0.0714285714285714, 0.0714285714285714},
                point_t{0.0714285714285714, 0.0714285714285714, 0.7857142857142857},
                point_t{0.0714285714285714, 0.7857142857142857, 0.0714285714285714},
                point_t{0.1005964238332008, 0.3994035761667992, 0.3994035761667992},
                point_t{0.3994035761667992, 0.1005964238332008, 0.3994035761667992},
                point_t{0.3994035761667992, 0.3994035761667992, 0.1005964238332008},
                point_t{0.3994035761667992, 0.1005964238332008, 0.1005964238332008},
                point_t{0.1005964238332008, 0.3994035761667992, 0.1005964238332008},
                point_t{0.1005964238332008, 0.1005964238332008, 0.3994035761667992},
            },
            .weights = {-0.0789333333333333, 0.0457333333333333, 0.0457333333333333,//
                        0.0457333333333333, 0.0457333333333333, 0.1493333333333333, //
                        0.1493333333333333, 0.1493333333333333, 0.1493333333333333, //
                        0.1493333333333333, 0.1493333333333333}};
    }
};

template<>
struct template_geometry_t<Tetrahedron, 6> : public base_template_geometry_t<Tetrahedron> {
    CUDA_CALLABLE static constexpr auto quadrature_info() {
        return quadrature_info_t<dim, 24>{
            .alg_acc = 6,
            .pnts = {
                point_t{0.3561913862225449, 0.2146028712591517, 0.2146028712591517},
                point_t{0.2146028712591517, 0.2146028712591517, 0.2146028712591517},
                point_t{0.2146028712591517, 0.2146028712591517, 0.3561913862225449},
                point_t{0.2146028712591517, 0.3561913862225449, 0.2146028712591517},
                point_t{0.8779781243961660, 0.0406739585346113, 0.0406739585346113},
                point_t{0.0406739585346113, 0.0406739585346113, 0.0406739585346113},
                point_t{0.0406739585346113, 0.0406739585346113, 0.8779781243961660},
                point_t{0.0406739585346113, 0.8779781243961660, 0.0406739585346113},
                point_t{0.0329863295731731, 0.3223378901422757, 0.3223378901422757},
                point_t{0.3223378901422757, 0.3223378901422757, 0.3223378901422757},
                point_t{0.3223378901422757, 0.3223378901422757, 0.0329863295731731},
                point_t{0.3223378901422757, 0.0329863295731731, 0.3223378901422757},
                point_t{0.2696723314583159, 0.0636610018750175, 0.0636610018750175},
                point_t{0.0636610018750175, 0.2696723314583159, 0.0636610018750175},
                point_t{0.0636610018750175, 0.0636610018750175, 0.2696723314583159},
                point_t{0.6030056647916491, 0.0636610018750175, 0.0636610018750175},
                point_t{0.0636610018750175, 0.6030056647916491, 0.0636610018750175},
                point_t{0.0636610018750175, 0.0636610018750175, 0.6030056647916491},
                point_t{0.0636610018750175, 0.2696723314583159, 0.6030056647916491},
                point_t{0.2696723314583159, 0.6030056647916491, 0.0636610018750175},
                point_t{0.6030056647916491, 0.0636610018750175, 0.2696723314583159},
                point_t{0.0636610018750175, 0.6030056647916491, 0.2696723314583159},
                point_t{0.2696723314583159, 0.0636610018750175, 0.6030056647916491},
                point_t{0.6030056647916491, 0.2696723314583159, 0.0636610018750175},
            },
            .weights = {0.0399227502581679, 0.0399227502581679, 0.0399227502581679,//
                        0.0399227502581679, 0.0100772110553207, 0.0100772110553207,//
                        0.0100772110553207, 0.0100772110553207, 0.0553571815436544,//
                        0.0553571815436544, 0.0553571815436544, 0.0553571815436544,//
                        0.0482142857142857, 0.0482142857142857, 0.0482142857142857,//
                        0.0482142857142857, 0.0482142857142857, 0.0482142857142857,//
                        0.0482142857142857, 0.0482142857142857, 0.0482142857142857,//
                        0.0482142857142857, 0.0482142857142857, 0.0482142857142857}};
    }
};

template<>
struct template_geometry_t<Tetrahedron, 8> : public base_template_geometry_t<Tetrahedron> {
    CUDA_CALLABLE static constexpr auto quadrature_info() {
        return quadrature_info_t<dim, 45>{
            .alg_acc = 8,
            .pnts = {
                point_t{0.2500000000000000, 0.2500000000000000, 0.2500000000000000},
                point_t{0.6175871903000830, 0.1274709365666390, 0.1274709365666390},
                point_t{0.1274709365666390, 0.1274709365666390, 0.1274709365666390},
                point_t{0.1274709365666390, 0.1274709365666390, 0.6175871903000830},
                point_t{0.1274709365666390, 0.6175871903000830, 0.1274709365666390},
                point_t{0.9037635088221031, 0.0320788303926323, 0.0320788303926323},
                point_t{0.0320788303926323, 0.0320788303926323, 0.0320788303926323},
                point_t{0.0320788303926323, 0.0320788303926323, 0.9037635088221031},
                point_t{0.0320788303926323, 0.9037635088221031, 0.0320788303926323},
                point_t{0.4502229043567190, 0.0497770956432810, 0.0497770956432810},
                point_t{0.0497770956432810, 0.4502229043567190, 0.0497770956432810},
                point_t{0.0497770956432810, 0.0497770956432810, 0.4502229043567190},
                point_t{0.0497770956432810, 0.4502229043567190, 0.4502229043567190},
                point_t{0.4502229043567190, 0.0497770956432810, 0.4502229043567190},
                point_t{0.4502229043567190, 0.4502229043567190, 0.0497770956432810},
                point_t{0.3162695526014501, 0.1837304473985499, 0.1837304473985499},
                point_t{0.1837304473985499, 0.3162695526014501, 0.1837304473985499},
                point_t{0.1837304473985499, 0.1837304473985499, 0.3162695526014501},
                point_t{0.1837304473985499, 0.3162695526014501, 0.3162695526014501},
                point_t{0.3162695526014501, 0.1837304473985499, 0.3162695526014501},
                point_t{0.3162695526014501, 0.3162695526014501, 0.1837304473985499},
                point_t{0.0229177878448171, 0.2319010893971509, 0.2319010893971509},
                point_t{0.2319010893971509, 0.0229177878448171, 0.2319010893971509},
                point_t{0.2319010893971509, 0.2319010893971509, 0.0229177878448171},
                point_t{0.5132800333608811, 0.2319010893971509, 0.2319010893971509},
                point_t{0.2319010893971509, 0.5132800333608811, 0.2319010893971509},
                point_t{0.2319010893971509, 0.2319010893971509, 0.5132800333608811},
                point_t{0.2319010893971509, 0.0229177878448171, 0.5132800333608811},
                point_t{0.0229177878448171, 0.5132800333608811, 0.2319010893971509},
                point_t{0.5132800333608811, 0.2319010893971509, 0.0229177878448171},
                point_t{0.2319010893971509, 0.5132800333608811, 0.0229177878448171},
                point_t{0.0229177878448171, 0.2319010893971509, 0.5132800333608811},
                point_t{0.5132800333608811, 0.0229177878448171, 0.2319010893971509},
                point_t{0.7303134278075384, 0.0379700484718286, 0.0379700484718286},
                point_t{0.0379700484718286, 0.7303134278075384, 0.0379700484718286},
                point_t{0.0379700484718286, 0.0379700484718286, 0.7303134278075384},
                point_t{0.1937464752488044, 0.0379700484718286, 0.0379700484718286},
                point_t{0.0379700484718286, 0.1937464752488044, 0.0379700484718286},
                point_t{0.0379700484718286, 0.0379700484718286, 0.1937464752488044},
                point_t{0.0379700484718286, 0.7303134278075384, 0.1937464752488044},
                point_t{0.7303134278075384, 0.1937464752488044, 0.0379700484718286},
                point_t{0.1937464752488044, 0.0379700484718286, 0.7303134278075384},
                point_t{0.0379700484718286, 0.1937464752488044, 0.7303134278075384},
                point_t{0.7303134278075384, 0.0379700484718286, 0.1937464752488044},
                point_t{0.1937464752488044, 0.7303134278075384, 0.0379700484718286},
            },
            .weights = {
                -0.2359620398477557,//
                0.0244878963560562, //
                0.0244878963560562, //
                0.0244878963560562, //
                0.0244878963560562, //
                0.0039485206398261, //
                0.0039485206398261, //
                0.0039485206398261, //
                0.0039485206398261, //
                0.0263055529507371, //
                0.0263055529507371, //
                0.0263055529507371, //
                0.0263055529507371, //
                0.0263055529507371, //
                0.0263055529507371, //
                0.0829803830550589, //
                0.0829803830550589, //
                0.0829803830550589, //
                0.0829803830550589, //
                0.0829803830550589, //
                0.0829803830550589, //
                0.0254426245481023, //
                0.0254426245481023, //
                0.0254426245481023, //
                0.0254426245481023, //
                0.0254426245481023, //
                0.0254426245481023, //
                0.0254426245481023, //
                0.0254426245481023, //
                0.0254426245481023, //
                0.0254426245481023, //
                0.0254426245481023, //
                0.0254426245481023, //
                0.0134324384376852, //
                0.0134324384376852, //
                0.0134324384376852, //
                0.0134324384376852, //
                0.0134324384376852, //
                0.0134324384376852, //
                0.0134324384376852, //
                0.0134324384376852, //
                0.0134324384376852, //
                0.0134324384376852, //
                0.0134324384376852, //
                0.0134324384376852,
            }};
    }
};

}// namespace vox::fields