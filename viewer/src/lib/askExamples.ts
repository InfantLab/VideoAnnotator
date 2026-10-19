/** Example questions for Ask AI (spec 027 R17): each says the answer's form. */
export interface ExampleQuestion {
  title: string;
  text: string;
}

export const EXAMPLE_QUESTIONS: ExampleQuestion[] = [
  {
    title: 'Touch',
    text:
      "Is the adult's hand or arm touching the infant's body? Overlap caused only by the camera angle is not touch.\n" +
      'Answer with one word: touching or not_touching.',
  },
  {
    title: 'Gaze',
    text: 'Where is the infant looking?\nAnswer with one word: at_caregiver, at_object or elsewhere.',
  },
  {
    title: 'Number of people',
    text: 'How many people can you see?\nAnswer with a number only.',
  },
  {
    title: 'Infant posture',
    text: "What is the infant's posture?\nAnswer with one word: lying, sitting, standing or held.",
  },
  {
    title: 'Activity',
    text: 'What is happening?\nAnswer with one of: feeding, playing, reading, changing, sleeping, other.',
  },
  {
    title: 'Describe',
    text: 'Describe what is happening in one sentence.',
  },
];
