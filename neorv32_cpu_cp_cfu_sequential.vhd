-- ================================================================================ --
-- NEORV32 CPU - Co-Processor: Custom (RISC-V Instructions) Functions Unit (CFU)    --
-- -------------------------------------------------------------------------------- --
-- Stateful TM clause accumulator.                                                  --
--                                                                                  --
-- R4-type instructions (custom-1 opcode):                                          --
--                                                                                  --
--   funct3 = 000  CHUNK_EVAL                                                       --
--     rs1 = x_bits (32 input bits for this chunk)                                  --
--     rs2 = pos_mask (32 positive literal include bits)                            --
--     rs3 = neg_mask (32 negative literal include bits)                            --
--     Updates internal clause_failed and clause_all_exclude registers.             --
--     result_o = 0 (ignored by caller)                                             --
--                                                                                  --
--   funct3 = 001  CLAUSE_COMMIT                                                    --
--     rs1(0) = clause polarity (0 = even = positive vote, 1 = odd = negative vote) --
--     Finalizes current clause and adds vote to class_sum.                         --
--     Resets clause_failed and clause_all_exclude for next clause.                 --
--     result_o = 0 (ignored by caller)                                             --
--                                                                                  --
--   funct3 = 010  GET_SCORE                                                        --
--     No operands used.                                                            --
--     Returns clamped class_sum as signed 32-bit value.                            --
--     Resets class_sum to 0 for next class.                                        --
--                                                                                  --
-- All operations complete in one clock cycle (valid_o asserted one cycle           --
-- after start_i, result_o valid the cycle after valid_o).                          --
-- ================================================================================ --

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity neorv32_cpu_cp_cfu is
  port (
    clk_i    : in  std_ulogic;
    rstn_i   : in  std_ulogic;
    start_i  : in  std_ulogic;
    active_i : in  std_ulogic;
    rtype_i  : in  std_ulogic;
    funct3_i : in  std_ulogic_vector(2 downto 0);
    funct7_i : in  std_ulogic_vector(6 downto 0);
    rs1_i    : in  std_ulogic_vector(31 downto 0);
    rs2_i    : in  std_ulogic_vector(31 downto 0);
    rs3_i    : in  std_ulogic_vector(31 downto 0);
    result_o : out std_ulogic_vector(31 downto 0);
    valid_o  : out std_ulogic
  );
end neorv32_cpu_cp_cfu;

architecture neorv32_cpu_cp_cfu_rtl of neorv32_cpu_cp_cfu is

  constant r4type_c : std_ulogic := '1';

  -- Threshold for clamping (must match C #define THRESHOLD)
  constant THRESHOLD : integer := 50;

  -- Internal state registers
  signal clause_failed      : std_ulogic;               -- mismatch seen in current clause
  signal clause_all_exclude : std_ulogic;               -- no literals seen in current clause yet
  signal class_sum          : signed(7 downto 0);       -- accumulated vote sum, range -50..+50

  -- One-cycle delay register for valid_o
  signal valid_reg  : std_ulogic;
  signal result_reg : std_ulogic_vector(31 downto 0);

begin

  -- Output registers (result is valid one cycle after valid_o)
  result_o <= result_reg;
  valid_o  <= valid_reg;

  cfu_seq: process(clk_i, rstn_i)
    variable x_bits      : std_ulogic_vector(31 downto 0);
    variable pos_mask    : std_ulogic_vector(31 downto 0);
    variable neg_mask    : std_ulogic_vector(31 downto 0);
    variable mismatch    : std_ulogic;
    variable all_zero    : std_ulogic;
    variable vote        : signed(7 downto 0);
    variable new_sum     : signed(7 downto 0);
    constant zero_vec    : std_ulogic_vector(31 downto 0) := (others => '0');
  begin
    if rstn_i = '0' then
      clause_failed      <= '0';
      clause_all_exclude <= '1';
      class_sum          <= (others => '0');
      valid_reg          <= '0';
      result_reg         <= (others => '0');

    elsif rising_edge(clk_i) then
      -- Default: not valid unless we process an instruction
      valid_reg  <= '0';
      result_reg <= (others => '0');

      if (start_i = '1') and (rtype_i = r4type_c) then
        valid_reg <= '1';  -- all operations complete in one cycle

        case funct3_i is

          -- ----------------------------------------------------------------
          -- funct3 = 000: CHUNK_EVAL
          -- Update clause_failed and clause_all_exclude based on this chunk.
          -- ----------------------------------------------------------------
          when "000" =>
            x_bits   := rs1_i;
            pos_mask := rs2_i;
            neg_mask := rs3_i;

            -- Check for mismatch
            if ((pos_mask and (not x_bits)) /= zero_vec) or
               ((neg_mask and x_bits) /= zero_vec) then
              mismatch := '1';
            else
              mismatch := '0';
            end if;

            -- Check if chunk has any literals
            if (pos_mask or neg_mask) = zero_vec then
              all_zero := '1';
            else
              all_zero := '0';
            end if;

            -- Update clause state
            -- clause_failed: set on mismatch, never cleared until CLAUSE_COMMIT
            if mismatch = '1' then
              clause_failed <= '1';
            end if;

            -- clause_all_exclude: cleared when we see any literal
            if all_zero = '0' then
              clause_all_exclude <= '0';
            end if;

            -- result_o not used by caller for this operation
            result_reg <= (others => '0');

          -- ----------------------------------------------------------------
          -- funct3 = 001: CLAUSE_COMMIT
          -- Finalize clause vote and add to class_sum.
          -- rs1(0) = polarity: 0 = positive vote (+1), 1 = negative vote (-1)
          -- ----------------------------------------------------------------
          when "001" =>
            -- Clause output: 1 if not failed and not all_exclude
            if (clause_failed = '0') and (clause_all_exclude = '0') then
              -- Determine vote direction from clause polarity (rs1 bit 0)
              if rs1_i(0) = '0' then
                vote := to_signed(1, 8);   -- even clause: +1
              else
                vote := to_signed(-1, 8);  -- odd clause:  -1
              end if;

              -- Add vote with clamping
              new_sum := class_sum + vote;

              if new_sum > to_signed(THRESHOLD, 8) then
                class_sum <= to_signed(THRESHOLD, 8);
              elsif new_sum < to_signed(-THRESHOLD, 8) then
                class_sum <= to_signed(-THRESHOLD, 8);
              else
                class_sum <= new_sum;
              end if;
            end if;
            -- else: clause failed or all_exclude, no vote added

            -- Reset clause state for next clause
            clause_failed      <= '0';
            clause_all_exclude <= '1';

            result_reg <= (others => '0');

          -- ----------------------------------------------------------------
          -- funct3 = 010: GET_SCORE
          -- Return clamped class_sum and reset for next class.
          -- ----------------------------------------------------------------
          when "010" =>
            -- Sign-extend class_sum to 32 bits
            result_reg <= std_ulogic_vector(resize(class_sum, 32));

            -- Reset for next class
            class_sum          <= (others => '0');
            clause_failed      <= '0';
            clause_all_exclude <= '1';

          -- ----------------------------------------------------------------
          -- funct3 = 011: CHUNK_EVAL_FAST
          -- Same as CHUNK_EVAL but returns clause_failed in bit 0 so C
          -- can break out of the chunk loop early.
          -- rs1 = x_bits, rs2 = pos_mask, rs3 = neg_mask
          -- result bit 0 = clause_failed (1 = clause already failed, break)
          -- ----------------------------------------------------------------
          when "011" =>
            x_bits   := rs1_i;
            pos_mask := rs2_i;
            neg_mask := rs3_i;

            if ((pos_mask and (not x_bits)) /= zero_vec) or
               ((neg_mask and x_bits) /= zero_vec) then
              mismatch := '1';
            else
              mismatch := '0';
            end if;

            if (pos_mask or neg_mask) = zero_vec then
              all_zero := '1';
            else
              all_zero := '0';
            end if;

            if mismatch = '1' then
              clause_failed <= '1';
            end if;

            if all_zero = '0' then
              clause_all_exclude <= '0';
            end if;

            -- Return clause_failed (after update) in bit 0
            -- Use mismatch OR existing clause_failed
            if (mismatch = '1') or (clause_failed = '1') then
              result_reg(0) <= '1';
            else
              result_reg(0) <= '0';
            end if;
            result_reg(31 downto 1) <= (others => '0');

          when others =>
            null;

        end case;
      end if;
    end if;
  end process cfu_seq;

end neorv32_cpu_cp_cfu_rtl;