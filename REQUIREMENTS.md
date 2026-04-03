# Structured Requirements

## 1. Explainability

- The system shall display decision factors used in route prioritisation.
- The system shall display the weight assigned to each decision factor.
- The system shall provide justification for the priority ranking of each defect.
- The system shall show why each defect is prioritised for treatment.

## 2. Visualisation

- The system shall visualise traffic conditions on the route.
- The system shall visualise incidents affecting route outcomes.
- The system shall visualise peak-hour conditions, including school, work, and freight peaks.
- The system shall clearly represent route priority using visual indicators such as numbering or colour variation.
- The system shall display only defects that are included in the final route result.
- The system shall provide a clear and descriptive legend using meaningful labels rather than abstract labels such as P1 or P2.

## 3. Input Design

- The system shall support multiple input methods including manual entry and CSV dataset upload.
- The system shall support scalable input volumes, including at least 20 defects.
- The system shall provide a visible input interface for defect entry on the HTML page.
- The system should support search-assisted input or equivalent guided location entry in future versions.

## 4. Data Quality

- The system shall require precise location input such as coordinates or full addresses.
- The system shall avoid vague location descriptions where a specific road, segment, suburb, or coordinate can be shown.
- The system shall display defect locations using specific and interpretable location descriptions.

## 5. Usability And UX

- The system shall present time, distance, and cost per route segment.
- The system shall present estimated service time for each defect stop.
- The system shall present route information in a clear and interpretable format suitable for end users.
- The system shall avoid ambiguous wording such as "other roads" where a more specific description can be used.
- The system shall present decision explanations, route ordering, and traffic effects in a way that can be understood quickly by a non-technical user.
