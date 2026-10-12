"""Regression coverage for Agent component and full-direction extrema matching."""

import unittest

from m3s.doi_workflow import closest_row, parse_requirement, within_tolerance


class PrincipalMatchingTests(unittest.TestCase):
    def test_directional_extremum_can_create_hit(self):
        cases = [
            ("young_modulus", "Emax", "E2", 0.7, 0.4),
            ("shear_modulus", "Gmax", "G23", 0.05, 0.02),
            ("poisson_ratio", "vmin", "v13", -1.5, -0.2),
        ]
        for family, extreme, principal, target, value in cases:
            with self.subTest(family=family):
                requirement = dict(
                    family=family, direction="any", target=target, tolerance=0.01
                )
                rows = [
                    {
                        "structure_id": "paper_image",
                        "elastic_constants": {
                            "is_stable": True,
                            extreme: target,
                            principal: value,
                        },
                    }
                ]
                best = closest_row(rows, {"paper_image": "image.png"}, requirement)
                self.assertEqual(best["matched_field"], extreme)
                self.assertTrue(within_tolerance(best["absolute_error"], 0.01))
                del rows[0]["elastic_constants"][principal]
                self.assertEqual(closest_row(rows, {"paper_image": "image.png"}, requirement)["matched_field"], extreme)

    def test_explicit_extrema_are_accepted(self):
        for field in ["Emax", "Emin", "Gmax", "Gmin", "vmax", "vmin", "E_max", "νmin"]:
            with self.subTest(field=field):
                unit = "" if field.startswith(("v", "ν")) else " GPa"
                requirement = parse_requirement(f"{field} 0.7{unit},tolerance 0.01{unit}")
                self.assertEqual(requirement["direction"], field.replace("_", "").replace("ν", "v"))
        with self.assertRaises(ValueError):
            closest_row([], {}, dict(family="young_modulus", direction="G12", target=0.7))

    def test_units_explicit_component_and_valid_hit(self):
        requirement = parse_requirement("Young modulus E2 700 MPa,tolerance 10 MPa")
        self.assertAlmostEqual(requirement["target"], 0.7)
        self.assertEqual(requirement["tolerance"], 0.01)
        rows = [
            {
                "structure_id": "paper_image_unstable",
                "elastic_constants": {"is_stable": False, "E2": 0.7},
            },
            {
                "structure_id": "unrelated",
                "elastic_constants": {"is_stable": True, "E2": 0.7},
            },
            {
                "structure_id": "paper_image",
                "elastic_constants": {
                    "is_stable": True,
                    "E1": 0.7,
                    "E2": 0.705,
                    "Emax": 0.7,
                },
            },
        ]
        best = closest_row(rows, {"paper_image": "image.png"}, requirement)
        self.assertEqual(best["matched_field"], "E2")
        self.assertEqual(best["matched_value"], 0.705)
        self.assertTrue(
            within_tolerance(best["absolute_error"], requirement["tolerance"])
        )

    def test_every_principal_component_can_match(self):
        for family, fields, target in [
            ("young_modulus", ["E1", "E2", "E3"], 0.7),
            ("shear_modulus", ["G12", "G23", "G31"], 0.05),
            ("poisson_ratio", ["v12", "v13", "v23", "v21", "v31", "v32"], -1.5),
        ]:
            for field in fields:
                with self.subTest(field=field):
                    best = closest_row(
                        [
                            {
                                "structure_id": "s",
                                "elastic_constants": {"is_stable": True, field: target},
                            }
                        ],
                        {"s": "s.png"},
                        dict(family=family, direction="any", target=target),
                    )
                    self.assertEqual(best["matched_field"], field)
                    self.assertEqual(best["absolute_error"], 0)


if __name__ == "__main__":
    unittest.main()
