/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include "common.h"
#include <aspect/reconstructed_fault/boundary_contact.h>

namespace
{
  using namespace aspect;
  using namespace dealii;
  std::vector<FaultBoundaryFace<2>>
                                 square_faces ()
  {
    std::vector<FaultBoundaryFace<2>> result (4);
    const Point<2> points[4][2] =
    {
      { { 0, 0 }, { 0, 1 } }, { { 1, 0 }, { 1, 1 } }, { { 0, 0 }, { 1, 0 } }, { { 0, 1 }, { 1, 1 } }
    };
    for (unsigned int i = 0; i < 4; ++i)
      {
        result[i].boundary_id = i;
        result[i].vertices = { { points[i][0], points[i][1] } };
        result[i].cell_lower = { 0, 0 };
        result[i].cell_upper = { 1, 1 };
        result[i].inward_normal[i / 2] = i % 2 ? -1. : 1.;
      }
    return result;
  }
  PrescribedInitialFault<2>
  fault (std::initializer_list<Point<2>> vertices)
  {
    return { vertices, std::vector<double> (vertices.size (), .6) };
  }
}

TEST_CASE ("Prescribed boundary contacts retain identity, orientation and topology",
           "[fault_boundary_contact]")
{
  using namespace aspect::ReconstructedFaultUtilities;
  auto faces = square_faces ();
  CHECK (detect_boundary_contacts<2> ({ fault ({ { .2, .2 }, { .8, .8 } }) }, faces).empty ());
  CHECK (detect_boundary_contacts<2> ({ fault ({ { .2, 1e-8 }, { .8, .8 } }) }, faces).empty ());
  for (const bool reverse :
  {
    false, true
  })
  {
    auto line = fault ({ { .2, 0 }, { .5, .5 }, { .8, 1 } });
    if (reverse)
      std::reverse (line.vertices.begin (), line.vertices.end ());
    auto contacts = detect_boundary_contacts<2> ({ line }, faces);
    REQUIRE (contacts.size () == 2);
    qualify_boundary_contact_support<2> (contacts, { line }, faces, { .05, .05 }, { .05 });
    for (unsigned int i = 0; i < 2; ++i)
      {
        CHECK (contacts[i].fault_index == 0);
        CHECK (contacts[i].endpoint == i);
        CHECK (contacts[i].boundary_id == (reverse ? 3 - i : 2 + i));
        CHECK (contacts[i].unsupported_reason.empty ());
        CHECK (contacts[i].inward_tangent * contacts[i].inward_boundary_normal > 0.);
        CHECK (contacts[i].influence_length == Approx (.03));
      }
  }
  const auto transverse = fault ({ { .5, 0 }, { .5, 1 } });
  auto contacts = detect_boundary_contacts<2> ({ transverse }, faces);
  qualify_boundary_contact_support<2> (contacts, { transverse }, faces, { .1, .1 }, { .1 });
  for (const auto &contact : contacts)
    {
      CHECK (contact.influence_length == 0.);
      CHECK (contact.unsupported_reason.empty ());
    }
  const auto diagonal = fault ({ { 0, .2 }, { .8, 1 } });
  contacts = detect_boundary_contacts<2> ({ diagonal }, faces);
  REQUIRE (contacts.size () == 2);
  CHECK (contacts[0].boundary_id == 0);
  CHECK (contacts[1].boundary_id == 3);
  const std::vector<PrescribedInitialFault<2>> multiple
  = { fault ({ { .2, 0 }, { .3, .4 } }), fault ({ { .8, 0 }, { .7, .4 } }) };
  contacts = detect_boundary_contacts (multiple, faces);
  qualify_boundary_contact_support (contacts, multiple, faces, { .02, .02 }, { .02, .02 });
  REQUIRE (contacts.size () == 2);
  CHECK (contacts[0].fault_index == 0);
  CHECK (contacts[1].fault_index == 1);
  for (const auto &contact : contacts)
    CHECK (contact.unsupported_reason.empty ());
  // A boundary subdivision vertex must not create two contacts.
  auto half = faces[2];
  half.vertices[1] = { .5, 0 };
  faces[2].vertices[0] = { .5, 0 };
  faces.push_back (half);
  contacts = detect_boundary_contacts<2> ({ transverse }, faces);
  CHECK (contacts.size () == 2);
  for (auto &face : faces)
    if (face.boundary_id == 2)
      face.periodic = true;
  contacts = detect_boundary_contacts<2> ({ transverse }, faces);
  REQUIRE (contacts.size () == 1);
  CHECK (contacts[0].boundary_id == 3);
}

TEST_CASE ("Unsupported contacts are detected rather than approximated", "[fault_boundary_contact]")
{
  using namespace aspect::ReconstructedFaultUtilities;
  const auto faces = square_faces ();
  auto contacts = detect_boundary_contacts<2> ({ fault ({ { 0, 0 }, { .5, .5 } }) }, faces);
  REQUIRE (contacts.size () == 1);
  CHECK (contacts[0].unsupported_reason.find ("corner") != std::string::npos);
  contacts = detect_boundary_contacts<2> ({ fault ({ { .2, 0 }, { .8, 0 } }) }, faces);
  REQUIRE_FALSE (contacts.empty ());
  CHECK (contacts[0].unsupported_reason.find ("tangential") != std::string::npos);
  contacts = detect_boundary_contacts<2> ({ fault ({ { .2, -.2 }, { .4, .2 } }) }, faces);
  REQUIRE (contacts.size () == 1);
  CHECK (contacts[0].endpoint == dealii::numbers::invalid_unsigned_int);
  CHECK (contacts[0].unsupported_reason.find ("topology") != std::string::npos);
  auto curved = fault ({ { .3, 0 }, { .31, .01 }, { .7, .5 } });
  contacts = detect_boundary_contacts<2> ({ curved }, faces);
  qualify_boundary_contact_support<2> (contacts, { curved }, faces, { .1 }, { .1 });
  CHECK_FALSE (contacts[0].unsupported_reason.empty ());
  const std::vector<PrescribedInitialFault<2>> overlap
  = { fault ({ { .3, 0 }, { .6, .5 } }), fault ({ { .31, 0 }, { .8, .5 } }) };
  contacts = detect_boundary_contacts (overlap, faces);
  qualify_boundary_contact_support (contacts, overlap, faces, { .1, .1 }, { .1, .1 });
  for (const auto &contact : contacts)
    CHECK_FALSE (contact.unsupported_reason.empty ());
  // The second centerline misses the completion rectangle, but its diffuse
  // support reaches it. Reject without selecting the nearest branch.
  const std::vector<PrescribedInitialFault<2>> diffuse_overlap
  = { fault ({ { .5, 0 }, { .8, .4 } }), fault ({ { .442, .056 }, { .466, .088 } }) };
  contacts = detect_boundary_contacts (diffuse_overlap, faces);
  REQUIRE (contacts.size () == 1);
  qualify_boundary_contact_support (contacts, diffuse_overlap, faces, { .05 }, { .05, .05 });
  CHECK (contacts[0].unsupported_reason.find ("ambiguous") != std::string::npos);
  CHECK_THROWS_WITH (detect_boundary_contacts<3> ({}, {}), Catch::Matchers::Contains ("3D"));
}
