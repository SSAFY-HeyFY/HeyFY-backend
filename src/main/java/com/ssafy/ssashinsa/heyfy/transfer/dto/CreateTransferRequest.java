package com.ssafy.ssashinsa.heyfy.transfer.dto;

import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@NoArgsConstructor
public class CreateTransferRequest {
    private String depositAccountNo;
    private Long amount;
}
